__precompile__(false)
module PINNInfuserMod

using StableRNGs, ComponentArrays
using Optimization
using OptimizationOptimisers: ADAM
using SciMLBase, SciMLSensitivity
using Statistics
using Zygote
using OrdinaryDiffEq: Vern7
using Lux
#GPU
using DiffEqGPU
using CUDA, LuxCUDA
using Base.Threads
using MPI

import Pkg; Pkg.add("CUDA")
import Pkg; Pkg.add("LuxCUDA")
import Pkg; Pkg.add("MPI")

const CUDA_AVAILABLE = Ref(false)
function _try_load_cuda()
    try
        @eval using CUDA, LuxCUDA, DiffEqGPU
        CUDA_AVAILABLE[] = CUDA.functional()
        CUDA_AVAILABLE[] && @info "CUDA detected — running on GPU ($(CUDA.name(CUDA.device())))"
    catch e
        @warn "CUDA/LuxCUDA not available, falling back to CPU. ($e)"
    end
end

include("Losses.jl")
using .LossesMod

include("Parameters.jl")
using .ParametersMod: parameters

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

function PINN_Infuser_f(
    ode_problems::Vector{SciMLBase.ODEProblem},
    ode_params_list,
    nn::Lux.Chain,
    target_data::AbstractMatrix{Float64};
    nn_vars::Union{Nothing,Vector{Int}} = nothing,
    optimizer = ADAM,
    reltol::Float32 = 1e-6,
    abstol::Float32 = 1e-6,
    early_stopping::Bool = true,
    plotting::Bool = true
)::Tuple{Any,Any,Any}

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

    st = _to_device(st)

    p_NN = _to_device(
        Float32(parameters.initialisation) .* ComponentVector{Float32}(p_NN_raw)
    )

    n_states      = length(ode_problems[1].u0)
    corr_mask     = _build_correction_mask(nn_vars, n_states)
    nn_out_weight = Float32(parameters.nn_output_weight)
 
    call_count = Ref(0)
    log_buffer = String[]          # batch prints; flushed every callback
    
    function make_pinn_ode(idx)
        ode_problem = ode_problems[idx]
        ode_params  = ode_params_list[idx]
        # u0 on device as Float32 (SVector stripped)
        u0_vec = _to_device(Float32.(Vector(ode_problem.u0)))
 
        return function pinn_ode(u, p, t)
            nn_output = nn(u, p, st)[1]   # p = p_NN flows gradient
 
            du_physics = Zygote.ignore() do
                Float32.(_to_cpu(ode_problem.f(
                    Float32.(Vector(_to_cpu(u))), ode_params, t)))
            end
 
            # Use precomputed mask — no findfirst allocation in hot path
            correction = map(1:n_states) do i
                k = corr_mask[i]
                k == 0 ? zero(eltype(nn_output)) :
                         nn_out_weight * nn_output[k]
            end
 
            return _to_device(du_physics) .+ correction
        end
    end

    function predict_all_gpu(p_NN)
        # Build an ensemble of ODEProblems, each remade with its own u0/params
        base_prob = ode_problems[1]
 
        prob_func = (prob, i, repeat) -> begin
            u0_vec = _to_device(Float32.(Vector(ode_problems[i].u0)))
            ODEProblem(make_pinn_ode(i), u0_vec, ode_problems[i].tspan, p_NN)
        end
 
        ensemble = EnsembleProblem(
            ODEProblem(make_pinn_ode(1),
                       _to_device(Float32.(Vector(base_prob.u0))),
                       base_prob.tspan, p_NN);
            prob_func = prob_func
        )
 
        solve(ensemble, Vern7(),
              EnsembleGPUArray(CUDA.CUDABackend());
              trajectories = length(ode_problems),
              saveat       = Float32.(parameters.training_time),
              dtmax        = Float32(parameters.dtmax),
              reltol       = reltol,
              abstol       = abstol,
              # InterpolatingAdjoint is cheaper than QuadratureAdjoint on GPU
              sensealg     = InterpolatingAdjoint(autojacvec=ZygoteVJP()))
    end

    function predict_all_cpu(p_NN)
        sols = Vector{Any}(undef, length(ode_problems))
        @threads for idx in eachindex(ode_problems)
            pinn_ode = make_pinn_ode(idx)
            u0_vec   = Float32.(Vector(ode_problems[idx].u0))
            prob     = ODEProblem(pinn_ode, u0_vec,
                                  ode_problems[idx].tspan, p_NN)
            sols[idx] = solve(prob, Vern7();
                saveat   = Float32.(parameters.training_time),
                dtmax    = Float32(parameters.dtmax),
                reltol   = reltol,
                abstol   = abstol,
                sensealg = InterpolatingAdjoint(autojacvec=ZygoteVJP()))
        end
        return sols
    end

    predict_all = CUDA_AVAILABLE[] ? predict_all_gpu : predict_all_cpu

    last_ctxs = Ref{Any}(nothing)
 
    function build_ctx_all(p)
        sols = predict_all(p)
        # Threaded context construction (cheap, but avoids serial map)
        ctxs = Vector{Any}(undef, length(ode_problems))
        @threads for i in eachindex(ode_problems)
            sol     = CUDA_AVAILABLE[] ? sols[i] : sols[i]   # unified access
            sol_arr = Float32.(_to_cpu(Array(sol))')         # (T × n_states)
            pred_norm = (sol_arr .- U_MEAN_cpu') ./ U_STD_cpu'
            ctxs[i] = (
                sol            = sol,
                pred_mat       = sol_arr,
                pred_norm      = pred_norm,
                data_norm      = _to_cpu(data_norm),         # losses expect CPU
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
        return ctxs
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
                push!(log_buffer,
                    "  [fwd] call=$(call_count[]) loss=computing…")
            end
 
            # Differentiable loss (serial over ctxs — Zygote needs serial here)
            total_loss = sum(
                loss(parameters.active, ctx, parameters.config).total
                for ctx in ctxs
            ) / length(ctxs)
 
            Zygote.ignore() do
                push!(log_buffer,
                    "  [fwd] call=$(call_count[]) " *
                    "loss=$(round(Float64(total_loss), sigdigits=5))")
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
        println("\nIter $iter | avg_loss: $(round(avg_loss, sigdigits=6)) " *
                "| threads=$(nthreads()) | gpu=$(CUDA_AVAILABLE[])")
 
        # ── Early stopping ───────────────────────────────────────────────────
        if early_stopping && iter > parameters.early_stopping_start
            recent = losses[max(1, end-5) : end-1]
            if losses[end] > maximum(recent) ||
               losses[end] > minimum(recent) + 1f-3
                @info "Early stopping at iter $iter, loss=$(losses[end])"
                return true
            end
        end
 
        # Invalidate cache so next forward pass stores a fresh copy
        last_ctxs[] = nothing
        return false
    end
 
    # ── 11. Solve ─────────────────────────────────────────────────────────────
    @info "Starting optimisation | gpu=$(CUDA_AVAILABLE[]) | threads=$(nthreads())"
    trained_params = Optimization.solve(
        optprob,
        optimizer(parameters.lr),
        callback = callback,
        maxiters  = parameters.iterations,
    )
 
    return (trained_params.u, st, losses)
end
 
end # module PINNInfuserMod
 