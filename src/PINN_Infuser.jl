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
    ode_problem::SciMLBase.ODEProblem,
    ode_mat_base,
    ode_params,
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

    function pinn_ode!(du, u, p_NN, t) # modifies the original ODE by adding nn 
        nn_output = nn(u, p_NN, st)[1]
        ode_problem.f(du, u, ode_params, t)
        for (k, i) in enumerate(nn_vars)
            du[i] += parameters.nn_output_weight * nn_output[k]  
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
            dtmax=parameters.dtmax,
            reltol=reltol,
            abstol=abstol
        )
        if temp_sol.retcode != ReturnCode.Success
            @warn "ODE solve failed: $(temp_sol.retcode)"
        end
        return temp_sol
    end

    function compute_nn_contributions(u_mat, p_NN) #logs of the nn contributions to each variable at each time step
        nn_out = nn(reshape(u_mat', length(u_mat[1,:]), :), p_NN, st)[1]  # (n_outputs × time)
        return (parameters.nn_output_weight .* nn_out)'  # (time × n_outputs)
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
                        loss(parameters.active, ctx, parameters.config).total # this calculates loss with the Losses module
                    end,
        adtype
    )

    optprob = Optimization.OptimizationProblem(optf, p_NN)
    losses = Float64[]
    nn_history = Vector{Matrix{Float64}}()

    callback = function (state, l)
        u_curr = hasproperty(state, :u) ? state.u : state

        if parameters.working_on == "hpc"
            ctx = build_ctx(state) # Check if 'state' or 'current_p' is intended here
        elseif parameters.working_on == "local"
            ctx = build_ctx(u_curr)
        else
            @info "specify hpc or not"
            return false # Avoid falling through
        end

        result = loss(parameters.active, ctx, parameters.config)  
        push!(losses, result.total)  
        log_str = "Iter $(length(losses))"
        for k in keys(result)
            log_str *= " | $(k): $(round(result[k], sigdigits=6))"
        end
        println(log_str)
        flush(stdout)
        push!(nn_history, compute_nn_contributions(ctx.pred_mat, u_curr))

        # ── Live prediction vs data plot ──────────────────────────────────
        iter = length(losses)
        if plotting && iter % parameters.plot_every == 0
            t    = collect(training_steps)
            subplots = [
                begin
                    p = plot(t, ctx.pred_mat[:, j];
                        label     = "PINN",
                        lw        = 2,
                        xlabel    = "t",
                        ylims     = parameters.ylims[j]
                    )
                    plot!(p, t, ode_mat_base[:, j];   # ← add this
                        label = "ODE",
                        lw    = 2,
                        ls    = :dot,
                    )
                    plot!(p, t, target_data[:, j];
                        label = "Data",
                        lw    = 2,
                        ls    = :dash,
                    )
                    p
                end
                for j in 1:length(parameters.vars)
            ]

            fig = plot(subplots...;
                layout     = (3, 2),
                plot_title = "Iter $iter  |  loss = $(round(result.total, sigdigits=5))",
                size       = (360 * 2, 280 * 3),
            )
            display(fig)
        end
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

    return (trained_params.u, st, losses, nn_history) 
end # function
end # module