module PINNInfuser_new

using StableRNGs, ComponentArrays, LinearAlgebra
using Optimization, OptimizationOptimisers
using SciMLBase, SciMLSensitivity
using Statistics
using OrdinaryDiffEq: Vern7
using Lux
using Plots

include("/Applications/Desktop/CODE/PINNFuser.jl/src/Losses.jl")
using .Losses


export PINN_Infuser_new

function PINN_Infuser_new(
    ode_problem::SciMLBase.ODEProblem,
    ode_params,
    nn::Lux.Chain,
    training_steps::AbstractRange,
    target_data::AbstractMatrix{Float64};
    active::Vector{String},
    config::NamedTuple,
    nn_vars::Union{Nothing,Vector{Int}} = nothing,
    nn_output_weight::Float64 = 1.0,
    optimizer = ADAM,
    learning_rate::Float64 = 1e-3,
    reltol::Float64 = 1e-6,
    abstol::Float64 = 1e-6,
    dtmax = Inf,
    iters::Int = 1000,
    early_stopping::Bool = true,
    plot_every::Int = 1,
    rng::StableRNG = StableRNG(5958)
)::Tuple{Any,Any, Any, Any}

    U_MEAN = vec(mean(target_data, dims = 1))
    U_STD = vec(std(target_data, dims = 1)) .+ 1e-6
    data_norm = (target_data .- U_MEAN') ./ U_STD'

    p_NN, st = Lux.setup(rng, nn)
    p_NN = 1e-3 * ComponentVector{Float64}(p_NN)

    function pinn_ode!(du, u, p_NN, t)
        nn_output = nn(u, p_NN, st)[1]
        ode_problem.f(du, u, ode_params, t)
        for (k, i) in enumerate(nn_vars)
            du[i] += nn_output_weight * nn_output[k]  
        end    
    end

    function predict(p_NN)
        prob = ODEProblem(
            (du, u, p, t) -> pinn_ode!(du, u, p, t), 
            ode_problem.u0,
            ode_problem.tspan,
            p_NN,
        )
        temp_sol = solve(
            prob, 
            Vern7(),
            saveat=training_steps,
            dtmax=dtmax,
            reltol=reltol,
            abstol=abstol
        )
        return temp_sol
    end

    function compute_nn_contributions(u_mat, p_NN)
        n_time = size(u_mat, 1)
        nn_contrib = zeros(n_time, length(nn_vars))
        for (k, i) in enumerate(nn_vars)
            for ti in 1:n_time
                nn_contrib[ti, k] = nn_output_weight * nn(u_mat[ti, :], p_NN, st)[1][k]
            end
        end
        cycle_sums = vec(sum(nn_contrib, dims=1))  # length(nn_vars) vector
        return cycle_sums
    end

    function build_ctx(p)
        sol       = predict(p)
        sol_arr   = Array(sol)'             # (time × vars)
        pred_mat  = sol_arr
        pred_norm = (sol_arr .- U_MEAN') ./ U_STD'
        return (
            sol            = sol,
            pred_mat       = pred_mat,
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
        )
    end
    
    last_ctx = Ref{Any}(nothing) # to save ctx
    adtype = Optimization.AutoForwardDiff()
    optf = Optimization.OptimizationFunction(
        (x, p) ->   begin
                        ctx = build_ctx(x) # this builds the current prediction and so on
                        last_ctx[] = ctx   # saving it so that we dont calculate it again in callback
                        loss(active, ctx, config).total # this calls the loss module
                    end,
        adtype
    )

    optprob = Optimization.OptimizationProblem(optf, p_NN)
    losses = Float64[]
    nn_history = Vector{Matrix{Float64}}()

    callback = function (state, l)
        ctx    = last_ctx[]
        result = loss(active, ctx, config)   # NamedTuple, e.g. (data=..., physics=..., total=...)
        push!(losses, result.total)          # same evaluation as the sub-components
        log_str = "Iter $(length(losses))"
        for k in keys(result)
            log_str *= " | $(k): $(round(result[k], sigdigits=6))"
        end
        println(log_str)
        push!(nn_history, compute_nn_contributions(ctx.pred_mat, state.u))

        # ── Live prediction vs data plot ──────────────────────────────────
        iter = length(losses)
        ylims = [
            (0, 130),
            (4, 8),
            (50, 150),
            (20, 25),
            (0, 150),
            (0, 70)
        ]
        if iter % plot_every == 0
            t    = collect(training_steps)
            subplots = [
                begin
                    p = plot(t, ctx.pred_mat[:, j];
                        label     = "PINN",
                        lw        = 2,
                        xlabel    = "t",
                        ylims     = ylims[j]
                    )
                    plot!(p, t, target_data[:, j];
                        label = "Data",
                        lw    = 2,
                        ls    = :dash,
                    )
                    p
                end
                for j in 1:6
            ]

            fig = plot(subplots...;
                layout     = (3, 2),
                plot_title = "Iter $iter  |  loss = $(round(result.total, sigdigits=5))",
                size       = (360 * 2, 280 * 3),
            )
            display(fig)
        end
    
        if early_stopping && length(losses) > 100 # let it train for at least 100 iters
            recent_min = minimum(losses[(end-20):(end-1)]) 
            not_improving = losses[end] - recent_min < 1e-6 # stop either when the loss is not improving much
            increasing    = losses[end] > recent_min # or when the loss is increasing

            if not_improving || increasing
                println("Early stopping at iteration $(length(losses)) with loss $(losses[end])")
                return true
            end
        end
        return false

    end

    trained_params = Optimization.solve(
        optprob,
        optimizer(learning_rate),
        callback = callback,
        maxiters = iters,
    )

    return (trained_params.u, st, losses, nn_history) 
end # function
end # module