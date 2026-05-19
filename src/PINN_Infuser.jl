__precompile__(false)
module PINNInfuserMod

using StableRNGs, ComponentArrays, LinearAlgebra
using Optimization, OptimizationOptimisers
using SciMLBase, SciMLSensitivity
using Statistics
using StaticArrays
using Zygote
using OrdinaryDiffEq: Vern7
using Lux
using Plots

include("Losses.jl")
using .LossesMod

include("Parameters.jl")
using .ParametersMod: parameters

export PINN_Infuser_f

function PINN_Infuser_f(
    ode_problems::Vector{SciMLBase.ODEProblem},
    ode_params_list,
    nn::Lux.Chain,
    target_data::AbstractMatrix{Float64};
    nn_vars::Union{Nothing,Vector{Int}} = nothing,
    optimizer = ADAM,
    reltol::Float64 = 1e-6,
    abstol::Float64 = 1e-6,
    early_stopping::Bool = true,
    plotting::Bool = true
)::Tuple{Any,Any,Any}

    U_MEAN = vec(mean(target_data, dims=1))
    U_STD  = vec(std(target_data,  dims=1)) .+ 1e-6
    data_norm = (target_data .- U_MEAN') ./ U_STD'

    rng = StableRNG(parameters.seed)
    p_NN, st = Lux.setup(rng, nn)
    p_NN = parameters.initialisation * ComponentVector{Float64}(p_NN)
    # ── iteration counter visible to both optf and callback ──────────────
    grad_call_count = Ref(0)

    function predict_all(p_NN)
        sols = map(1:length(ode_problems)) do idx
            ode_problem = ode_problems[idx]
            ode_params  = ode_params_list[idx]
            u0_vec      = Vector{Float64}(ode_problem.u0)  # ← strip SVector here

            function pinn_ode(u, p, t)
                nn_output  = nn(u, p, st)[1]
                # physics doesn't depend on p_NN → ignore for gradient
                du_physics = Zygote.ignore() do
                    Vector{Float64}(ode_problem.f(Vector{Float64}(u), ode_params, t))
                end
                correction = [i in nn_vars ?
                            parameters.nn_output_weight * nn_output[findfirst(==(i), nn_vars)] :
                            zero(eltype(nn_output))
                            for i in 1:length(u0_vec)]
                return du_physics .+ correction
            end

            prob = ODEProblem(pinn_ode, u0_vec, ode_problem.tspan, p_NN)  # ← u0_vec not u0
            solve(prob, Vern7();
                saveat   = parameters.training_time,
                dtmax    = parameters.dtmax,
                reltol   = reltol,
                abstol   = abstol,
                sensealg = QuadratureAdjoint(autojacvec=ZygoteVJP()))
        end
        return sols
    end

    function build_ctx_all(p)
        sols = predict_all(p)
        ctxs = map(zip(sols, ode_problems, ode_params_list)) do (sol, ode_problem, ode_params)
            sol_arr   = Array(sol)'
            pred_norm = (sol_arr .- U_MEAN') ./ U_STD'
            (
                sol            = sol,
                pred_mat       = sol_arr,
                pred_norm      = pred_norm,
                data_norm      = data_norm,
                target_data    = target_data,
                training_steps = parameters.training_time,
                ode_params     = ode_params,
                ode_problem    = ode_problem,
                nn_vars        = nn_vars,
                p_NN           = p,
                nn             = nn,
                st             = st,
            )
        end
        return ctxs
    end

    
    last_ctxs       = Ref{Any}(nothing)
    call_count      = Ref(0)
    dual_call_count = Ref(0)
    
    adtype = Optimization.AutoZygote()

    optf = Optimization.OptimizationFunction(
        (x, p) -> begin
            ctxs = build_ctx_all(x)
            Zygote.ignore() do
                last_ctxs[] = ctxs
                call_count[] += 1
                println("  [forward] call=$(call_count[]) loss=computing...")
                flush(stdout)
            end
            total_loss = sum(loss(parameters.active, ctx, parameters.config).total
                            for ctx in ctxs) / length(ctxs)
            Zygote.ignore() do
                println("  [forward] call=$(call_count[]) loss=$(round(Float64(total_loss), sigdigits=5))")
                flush(stdout)
            end
            return total_loss
        end,
        adtype
    )
    optprob = Optimization.OptimizationProblem(optf, p_NN)
    losses  = Float64[]

    callback = function (state, l)
        if parameters.working_on == "hpc"
            ctxs = build_ctx_all(hasproperty(state, :u) ? state.u : state)
        elseif parameters.working_on == "local"
            ctxs = last_ctxs[]
            isnothing(ctxs) && return false
        else
            @info "specify hpc or local"
            return false
        end

        results  = [loss(parameters.active, ctx, parameters.config) for ctx in ctxs]
        avg_loss = mean(r.total for r in results)
        push!(losses, Float64(avg_loss))
        println("\nIter $(length(losses)) | avg_loss: $(round(avg_loss, sigdigits=6))")

        if early_stopping && length(losses) > parameters.early_stopping_start
            recent = losses[(end-5):(end-1)]
            if losses[end] > maximum(recent) || minimum(recent) + 1e-4 < losses[end]
                @info "Early stopping at iteration $(length(losses)) with loss $(losses[end])"
                return true
            end
        end
        return false
    end

    trained_params = Optimization.solve(
        optprob,
        optimizer(parameters.lr),
        callback = callback,
        maxiters  = parameters.iterations,
    )

    return (trained_params.u, st, losses)
end
end