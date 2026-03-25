module PINNInfuser_module

using StableRNGs, ComponentArrays, LinearAlgebra
using Optimization, OptimizationOptimisers
using SciMLBase, SciMLSensitivity
using Statistics
using ForwardDiff
using OrdinaryDiffEq: Vern7
using Lux
using Plots

include("Losses.jl")
using .Losses_module

export PINN_Infuser_funct

function PINN_Infuser_funct(
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
    p_NN = 1e-4 * ComponentVector{Float64}(p_NN)

    function pinn_ode!(du, u, p_NN, t) # modifies the original ODE by adding nn 
        nn_output = nn(u, p_NN, st)[1]
        ode_problem.f(du, u, ode_params, t)
        for (k, i) in enumerate(nn_vars)
            du[i] += nn_output_weight * nn_output[k]  
        end    
    end

    function predict(p_NN) # solving the modified ODE, outputs the new 10 states 
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

    function compute_nn_contributions(u_mat, p_NN) #logs of the nn contributions to each variable at each time step
        nn_out = nn(reshape(u_mat', length(u_mat[1,:]), :), p_NN, st)[1]  # (n_outputs × time)
        return (nn_output_weight .* nn_out)'  # (time × n_outputs)
    end

    function build_ctx(p)
        sol       = predict(p)              # predicted plots with nn additions based on loss reduction
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
    
    adtype = Optimization.AutoForwardDiff()
    optf = Optimization.OptimizationFunction(
        (x, p) ->   begin
                        ctx = build_ctx(x) # this builds the current prediction, parameters, state and so on
                        loss(active, ctx, config).total # this calculates loss with the Losses module
                    end,
        adtype
    )

    optprob = Optimization.OptimizationProblem(optf, p_NN)
    losses = Float64[]
    nn_history = Vector{Matrix{Float64}}()

    callback = function (state, l)
        ctx    = build_ctx(state.u)
        result = loss(active, ctx, config)  
        push!(losses, result.total)  
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
        # ── Early stopping ────────────────────────────────────────────────
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