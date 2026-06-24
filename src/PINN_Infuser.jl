__precompile__(false)
module PINNInfuserMod

using StableRNGs, ComponentArrays
using Optimization
using OptimizationOptimisers: Adam
using SciMLBase, SciMLSensitivity
using Statistics
using Zygote
using OrdinaryDiffEq: Vern7
using Lux
using Base.Threads

include("Parameters.jl")
using .ParametersMod: parameters

if parameters.working_on == "hpc"
    using Lux, LuxCUDA, CUDA
    const CUDA_AVAILABLE = Ref(false)
    function _try_load_cuda()
        try
            @eval using CUDA
            CUDA.allowscalar(false)

            if CUDA.functional(true)
                CUDA_AVAILABLE[] = true
                @info "CUDA functional"
            else
                CUDA_AVAILABLE[] = false
                @warn "CUDA installed but not functional"
            end

        catch err
            CUDA_AVAILABLE[] = false
            @warn "CUDA/LuxCUDA not available, falling back to CPU." err
        end
    end

    include("Losses.jl")
    using .LossesMod: loss

    export PINN_Infuser_f

    _to_device(x) = CUDA_AVAILABLE[] ? CUDA.cu(x) : x
    _to_cpu(x)    = CUDA_AVAILABLE[] ? Array(x)   : x

    function _build_correction_mask(nn_vars, n_states)
        mask = zeros(Int, n_states)
        if !isnothing(nn_vars)
            for (k, i) in enumerate(nn_vars)
                mask[i] = k
            end
        end
        return mask
    end
end 

function PINN_Infuser_f(
    ode_problems::Vector{SciMLBase.ODEProblem},
    ode_params_list,
    nn::Lux.Chain,
    target_data::AbstractMatrix{Float64};
    nn_vars::Union{Nothing,Vector{Int}} = nothing,
    optimizer = Adam,
    reltol::Real = 1e-6,
    abstol::Real = 1e-6,
    early_stopping::Bool = true,
    plotting::Bool = true
)::Tuple{Any,Any,Any}
    if parameters.working_on == "hpc"
        _try_load_cuda()

        U_MEAN_cpu = vec(mean(target_data, dims=1))
        U_STD_cpu  = vec(std(target_data,  dims=1)) .+ 1f-6
        # Cast to Float32 and push to device
        target_f32  = Float32.(target_data)
        U_MEAN      = _to_device(Float32.(U_MEAN_cpu))
        U_STD       = _to_device(Float32.(U_STD_cpu))
        data_norm   = _to_device((target_f32 .- U_MEAN_cpu') ./ U_STD_cpu')

        rng  = StableRNG(parameters.seed)
        p_NN_raw, st = Lux.setup(rng, nn)
        
        st = p_NN_raw  |> cpu_device()   
        p_NN = Float32(parameters.initialisation) .* ComponentVector{Float32}(p_NN_raw)

        n_states      = length(ode_problems[1].u0)
        corr_mask     = _build_correction_mask(nn_vars, n_states)
        nn_out_weight = Float32(parameters.nn_output_weight)

        call_count = Ref(0)
        log_buffer = String[]          # batch prints; flushed every callback

        function predict_all(p_NN)
            map(1:length(ode_problems)) do idx # length(ode_problems) is the number of patients
                prob_i   = ode_problems[idx]
                params_i = ode_params_list[idx]
                u0_vec   = Float32.(Vector(prob_i.u0))

                function pinn_ode(u, p, t)
                    nn_out = Float64.(nn(Float32.(u), p, st)[1])
                    du_physics = Zygote.ignore() do
                        Vector{Float64}(prob_i.f(Float64.(u), params_i, t))
                    end
                    correction = [i in nn_vars ?
                                Float64(nn_out_weight) * nn_out[findfirst(==(i), nn_vars)] :
                                zero(eltype(p))
                                for i in 1:length(u0_vec)]
                    return du_physics .+ correction
                end
                solve(ODEProblem(pinn_ode, u0_vec, prob_i.tspan, p_NN),
                    Vern7();
                    saveat   = parameters.training_time,
                    dtmax    = parameters.dtmax,
                    reltol   = 1e-6,
                    abstol   = 1e-6,
                    sensealg = InterpolatingAdjoint(autojacvec=ZygoteVJP()))
            end
        end
            
        last_ctxs = Ref{Any}(nothing)

        function build_ctx_all(p)
            sols = predict_all(p)
            map(1:length(ode_problems)) do i
                sol_arr = Float32.(Array(sols[i])')   # (T × n_states)
                pred_norm        = (sol_arr .- U_MEAN_cpu') ./ U_STD_cpu'
                patient_data_norm = (target_f32 .- U_MEAN_cpu') ./ U_STD_cpu'
                (
                    sol            = sols[i],
                    pred_mat       = sol_arr,
                    pred_norm      = pred_norm,
                    data_norm      = patient_data_norm,
                    target_data    = target_f32,
                    training_steps = parameters.training_time,
                    ode_params     = ode_params_list[i],
                    ode_problem    = ode_problems[i],
                    nn_vars        = nn_vars,
                    p_NN           = p,
                    nn             = nn,
                    st             = st,
                )
            end
        end
        
        # ── 8. Threaded loss reduction helper ────────────────────────────────────
        function compute_avg_loss(ctxs, p)
            n = length(ctxs)
            loss_vals = Vector{Float32}(undef, n)
            @threads for i in 1:n
                loss_vals[i] = Float32(
                    loss(parameters.active, ctxs[i], parameters.config).total
                )
            end
            return mean(loss_vals)
        end

        # ── 9. Optimization problem ───────────────────────────────────────────────
        adtype = Optimization.AutoZygote()

        optf = Optimization.OptimizationFunction(
            (x, _p) -> begin
                ctxs = build_ctx_all(x)

                Zygote.ignore() do
                    last_ctxs[]  = ctxs
                    call_count[] += 1
                    push!(log_buffer)
                end

                # Differentiable loss (serial over ctxs — Zygote needs serial here)
                total_loss = sum(
                    loss(parameters.active, ctx, parameters.config).total
                    for ctx in ctxs
                ) / length(ctxs)

                Zygote.ignore() do
                    println("\ncall $(call_count[]) ")
                end

                return total_loss
            end,
            adtype
        )

        optprob = Optimization.OptimizationProblem(optf, p_NN)
        losses  = Float32[]

        # ── 10. Callback ─────────────────────────────────────────────────────────
        callback = function (state, l)
            # Flush buffered log lines once per callback (much cheaper I/O)
            Zygote.ignore() do
                foreach(println, log_buffer)
                empty!(log_buffer)
                flush(stdout)
            end

            # Re-use cached ctxs on HPC (avoids a second full forward pass)
            ctxs = if parameters.working_on == "hpc"
                # On HPC the optimiser may run the callback off-thread; cache is
                # safest but we fall back to a fresh build if stale.
                isnothing(last_ctxs[]) ?
                    build_ctx_all(hasproperty(state, :u) ? state.u : state) :
                    last_ctxs[]
            elseif parameters.working_on == "local"
                isnothing(last_ctxs[]) && return false
                last_ctxs[]
            else
                @info "parameters.working_on must be \"hpc\" or \"local\""
                return false
            end

            # Threaded loss evaluation in the callback (no gradient needed)
            avg_loss = compute_avg_loss(ctxs, hasproperty(state, :u) ? state.u : state)
            push!(losses, Float32(avg_loss))

            iter = length(losses)
            println("Iter $iter | avg_loss: $(round(Float64(avg_loss), sigdigits=5))")
    
            # ── Early stopping ───────────────────────────────────────────────────
            if early_stopping && iter > parameters.early_stopping_start
                recent = losses[max(1, end-5) : end-1]
                if losses[end] > maximum(recent) 
                    @info "Early stopping because of increasing loss at iter $iter, loss=$(losses[end])"
                    return true
                elseif losses[end] > minimum(recent) + parameters.min_loss
                    @info "Early stopping because of reaching minimum at iter $iter, loss=$(losses[end])"
                    return true
                end
            end

            # ── Save model every 100 iterations ─────────────────────────────────────────────
            if iter % 100 == 0
                @info "Saving model at iteration $iter"
                current_params = hasproperty(state, :u) ? state.u : state   # ← already doing this elsewhere
                jldsave(parameters.savepath;
                        trained_p = current_params,
                        trained_st = st,
                        losses = losses
                        )
            end
            
            # Invalidate cache so next forward pass stores a fresh copy
            last_ctxs[] = nothing
            return false
        end

        # ── 11. Solve ─────────────────────────────────────────────────────────────
        @info "Starting optimisation"
        @info "Active losses = $(parameters.active)"
        trained_params = Optimization.solve(
            optprob,
            optimizer(parameters.lr),
            callback = callback,
            maxiters  = parameters.iterations,
        )

        return (trained_params.u, st, losses)
    end
end

end # module PINNInfuserMod

