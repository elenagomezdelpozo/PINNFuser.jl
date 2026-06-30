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
using DataInterpolations: LinearInterpolation, ExtrapolationType
using Base.Threads
using JLD2

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
    ode_problem::SciMLBase.ODEProblem,                      # single generic ODE
    ode_params,                                              # single param set
    nn::Lux.Chain,
    target_data_list::Vector{<:AbstractMatrix{Float64}};    # one (T × n_states) matrix per patient
    nn_vars::Union{Nothing,Vector{Int}} = nothing,
    optimizer = Adam,
    reltol::Real = 1e-6,
    abstol::Real = 1e-6,
    early_stopping::Bool = true,
    plotting::Bool = true
)::Tuple{Any,Any,Any}

    if parameters.working_on == "hpc"
        _try_load_cuda()

        n_patients = length(target_data_list)
        n_states   = length(ode_problem.u0)
        nn_out_weight = Float32(parameters.nn_output_weight)

        # ── 1. Global normalisation across all patients ───────────────────────────
        all_data_cat = vcat([Float32.(d) for d in target_data_list]...)  # (T*P × n_states)
        U_MEAN_cpu   = vec(mean(all_data_cat, dims=1))                    # (n_states,)
        U_STD_cpu    = vec(std(all_data_cat,  dims=1)) .+ 1f-6

        # Per-patient Float32 targets — kept on CPU for interpolation inside ODE
        target_f32_list = [Float32.(d) for d in target_data_list]        # Vector of (T × n_states)

        # ── 2. Build per-patient DataInterpolations objects (CPU, Zygote-invisible)
        # These give u_obs(t) at any t the adaptive solver requests.
        # One interpolant per state per patient: itp_list[p][s](t) → Float32
        t_obs = Float32.(parameters.training_time)   # (T,) observed timepoints

        itp_list = map(target_f32_list) do data_p   # data_p: (T × n_states)
            map(1:n_states) do s
                LinearInterpolation(new_patient_data[:, s], t_obs; extrapolation = ExtrapolationType.Extension)
            end
        end

        # ── 3. NN setup ───────────────────────────────────────────────────────────
        rng          = StableRNG(parameters.seed)
        p_NN_raw, st = Lux.setup(rng, nn)
        st           = p_NN_raw |> cpu_device()
        p_NN         = Float32(parameters.initialisation) .* ComponentVector{Float32}(p_NN_raw)

        call_count = Ref(0)
        log_buffer = String[]

        # ── 4. Per-patient ODE solve ──────────────────────────────────────────────
        # For each patient p, the ODE derivative at time t is:
        #   du = f(u_sim, params, t)  +  nn_out_weight * NN([u_obs(t); u_sim(t)], p_NN)
        # The NN input is 2*n_states wide; the interpolation is Zygote.ignore()'d
        # so it doesn't pollute the gradient tape.
        function predict_patient(p_NN, patient_idx)
            itps_p = itp_list[patient_idx]
            u0_vec = Float32.(Vector(ode_problem.u0))

            function pinn_ode(u, p, t)
                # u_obs(t) via interpolation — invisible to Zygote
                u_obs = Zygote.ignore() do
                    Float32[itps_p[s](t) for s in 1:n_states]
                end

                # NN input: concatenate observed and simulated states
                nn_input = vcat(u_obs, Float32.(u))          # (2*n_states,)
                nn_out   = Float64.(nn(nn_input, p, st)[1])  # (length(nn_vars),)

                # Physics derivative (Zygote-invisible — no params to differentiate)
                du_physics = Zygote.ignore() do
                    Vector{Float64}(ode_problem.f(Float64.(u), ode_params, t))
                end

                # Apply corrections only to nn_vars
                correction = [i in nn_vars ?
                              Float64(nn_out_weight) * nn_out[findfirst(==(i), nn_vars)] :
                              zero(eltype(p))
                              for i in 1:length(u0_vec)]

                return du_physics .+ correction
            end

            solve(
                ODEProblem(pinn_ode, u0_vec, ode_problem.tspan, p_NN),
                Vern7();
                saveat   = parameters.training_time,
                dtmax    = parameters.dtmax,
                reltol   = reltol,
                abstol   = abstol,
                sensealg = InterpolatingAdjoint(autojacvec=ZygoteVJP())
            )
        end

        # ── 5. Build context for all patients ─────────────────────────────────────
        last_ctxs = Ref{Any}(nothing)

        function build_ctx_all(p)
            map(1:n_patients) do i
                sol     = predict_patient(p, i)
                sol_arr = Float32.(Array(sol)')              # (T × n_states)

                pred_norm        = (sol_arr        .- U_MEAN_cpu') ./ U_STD_cpu'
                data_norm        = (target_f32_list[i] .- U_MEAN_cpu') ./ U_STD_cpu'

                (
                    sol            = sol,
                    pred_mat       = sol_arr,
                    pred_norm      = pred_norm,
                    data_norm      = data_norm,
                    target_data    = target_f32_list[i],
                    training_steps = parameters.training_time,
                    ode_params     = ode_params,
                    ode_problem    = ode_problem,
                    nn_vars        = nn_vars,
                    p_NN           = p,
                    nn             = nn,
                    st             = st,
                )
            end
        end

        # ── 6. Threaded loss reduction (callback only — no gradient) ──────────────
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

        # ── 7. Optimisation function ──────────────────────────────────────────────
        adtype = Optimization.AutoZygote()

        optf = Optimization.OptimizationFunction(
            (x, _p) -> begin
                ctxs = build_ctx_all(x)

                Zygote.ignore() do
                    last_ctxs[]   = ctxs
                    call_count[] += 1
                end

                # Serial sum over patients — Zygote requires serial here
                total_loss = sum(
                    loss(parameters.active, ctx, parameters.config).total
                    for ctx in ctxs
                ) / n_patients

                Zygote.ignore() do
                    println("\ncall $(call_count[])")
                end

                return total_loss
            end,
            adtype
        )

        optprob = Optimization.OptimizationProblem(optf, p_NN)
        losses  = Float32[]

        # ── 8. Callback ───────────────────────────────────────────────────────────
        callback = function (state, l)
            Zygote.ignore() do
                foreach(println, log_buffer)
                empty!(log_buffer)
                flush(stdout)
            end

            ctxs = if parameters.working_on == "hpc"
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

            current_params = hasproperty(state, :u) ? state.u : state
            avg_loss = compute_avg_loss(ctxs, current_params)
            push!(losses, Float32(avg_loss))

            iter = length(losses)
            println("Iter $iter | avg_loss: $(round(Float64(avg_loss), sigdigits=5))")

            # ── Early stopping ───────────────────────────────────────────────────
            if early_stopping && iter > parameters.early_stopping_start
                recent = losses[max(1, end-5) : end-1]
                if losses[end] > maximum(recent)
                    @info "Early stopping at iter $iter — loss increasing"
                    return true
                elseif losses[end] > minimum(recent) + parameters.min_loss
                    @info "Early stopping at iter $iter — loss plateaued"
                    return true
                end
            end

            # ── Checkpoint ───────────────────────────────────────────────────────
            if iter % 100 == 0
                @info "Saving model at iteration $iter"
                current_params = hasproperty(state, :u) ? state.u : state   # ← already doing this elsewhere
                jldsave(parameters.savepath;
                        trained_p = current_params,
                        trained_st = st,
                        losses = losses
                        )
            end

            last_ctxs[] = nothing
            return false
        end

        # ── 9. Solve ──────────────────────────────────────────────────────────────
        @info "Starting optimisation — $(n_patients) patients, single generic ODE"
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

