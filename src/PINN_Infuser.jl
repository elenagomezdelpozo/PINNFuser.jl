__precompile__(false)  # Add this here
module PINNInfuserMod

using StableRNGs, ComponentArrays, LinearAlgebra
using Optimization, OptimizationOptimisers
using SciMLBase, SciMLSensitivity
using Statistics
using ForwardDiff
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
    training_steps::AbstractRange,
    target_data::AbstractMatrix{Float64};
    nn_vars::Union{Nothing,Vector{Int}} = nothing,
    optimizer = ADAM,
    reltol::Float64 = 1e-6,
    abstol::Float64 = 1e-6,
    early_stopping::Bool = true,
    plotting::Bool = true,
    rng::StableRNG = StableRNG(5958)
)::Tuple{Any,Any, Any, Any}

    U_MEAN = vec(mean(target_data, dims = 1))
    U_STD = vec(std(target_data, dims = 1)) .+ 1e-6
    data_norm = (target_data .- U_MEAN') ./ U_STD'

    p_NN, st = Lux.setup(rng, nn)
    p_NN = parameters.initialisation * ComponentVector{Float64}(p_NN)

    function predict_all(p_NN) # solving the modified ODE, outputs the new 10 states 
        sols = []
        for (ode_problem, ode_params) in zip(ode_problems, ode_params_list)

            function pinn_ode!(du, u, p, t)
                nn_output = nn(u, p, st)[1]
                ode_problem.f(du, u, ode_params, t)
                for (k, i) in enumerate(nn_vars)
                    du[i] += parameters.nn_output_weight * nn_output[k]
                end
            end

            prob = ODEProblem(
                pinn_ode!,
                ode_problem.u0,
                ode_problem.tspan,
                p_NN
            )

            sol = solve(prob, Vern7();
                saveat = training_steps,
                dtmax  = parameters.dtmax,
                reltol = reltol,
                abstol = abstol
            )

            push!(sols, sol)
        end
        return sols
    end

    function build_ctx_all(p)
        sols = predict_all(p)
        ctxs = []
        for (sol, ode_problem, ode_params) in zip(
            sols, ode_problems, ode_params_list
        )
            sol_arr = Array(sol)'
            pred_norm = (sol_arr .- U_MEAN') ./ U_STD'
            push!(ctxs, (
                sol            = sol,
                pred_mat       = sol_arr,
                pred_norm      = pred_norm,
                data_norm      = data_norm,
                target_data    = target_data,
                training_steps = training_steps,
                ode_params     = ode_params,
                ode_problem    = ode_problem,
                nn_vars        = nn_vars,
                p_NN           = p,
                nn             = nn,
                st             = st,
            ))
        end
        return ctxs
    end
    
    adtype = Optimization.AutoForwardDiff()
    optf = Optimization.OptimizationFunction(
        (x, p) -> begin
            ctxs = build_ctx_all(x)

            total_loss = 0.0
            for ctx in ctxs
                total_loss += loss(parameters.active, ctx, parameters.config).total
            end

            return total_loss / length(ctxs)
        end,
        adtype
    )

    optprob = Optimization.OptimizationProblem(optf, p_NN)
    losses = Float64[]

    callback = function (state, l)
        u_curr = hasproperty(state, :u) ? state.u : state

        if parameters.working_on == "hpc"
            ctxs = build_ctx(state) # Check if 'state' or 'current_p' is intended here
        elseif parameters.working_on == "local"
            ctxs = build_ctx(u_curr)
        else
            @info "specify hpc or not"
            return false # Avoid falling through
        end

        results = [loss(parameters.active, ctx, parameters.config) for ctx in ctxs]
        avg_loss = mean(r.total for r in results)
        push!(losses, avg_loss)
        log_str = "Iter $(length(losses)) | avg_loss: $(round(avg_loss, sigdigits=6))"
        println(log_str)
        # ── Early stopping ────────────────────────────────────────────────
        if early_stopping && (length(losses) > parameters.early_stopping_start) # let it train for at least x iters
            recent_min = minimum(losses[(end-5):(end-1)]) # window of improvement is last 5 iterations
            not_improving = recent_min + 1e-4 < losses[end] # not improving much in the window (less than 1e-5 better)
            recent_max = maximum(losses[(end-5):(end-1)]) # window of improvement is last 5 iterations
            increasing    = losses[end] > recent_max # or when the loss is increasing compared to the last 5 iters
 
            if not_improving || increasing
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
        maxiters = parameters.iterations,
    )

    return (trained_params.u, st, losses) 
end # function
end # module