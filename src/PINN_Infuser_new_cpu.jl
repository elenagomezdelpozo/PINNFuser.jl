module PINNInfuser_new

using Lux, StableRNGs, Optimization, OptimizationOptimisers, ComponentArrays, LinearAlgebra
using OrdinaryDiffEq, Statistics, ForwardDiff
using OrdinaryDiffEq: Tsit5, Rodas4
using Printf
using Plots
using Lux, Plots, Zygote, Statistics, StableRNGs, ComponentArrays
using ForwardDiff
export PINN_Infuser_new

"""

Trains a Physics-Informed Neural Network (PINN) by minimizing a composite loss function
that includes both data fidelity and physical law adherence.

# Arguments
- `ode_problem::SciMLBase.ODEProblem`: The ODE problem defining the physical laws.
- `nn::Lux.Chain`: The Lux neural network model to be trained.
- `training_steps::AbstractRange`: The time steps at which to evaluate the solution and calculate loss function.
- `target_data::Array{Float64}`: The ground truth data for training.

# Keyword Arguments
- `early_stopping::Bool = true`: Whether to enable early stopping based on loss convergence.
- `nn_output_weight::Float64 = 0.1`: The weight factor for the NN infusion in ODE.
- `physics_weight::Float64 = 1.0`: The weight of the physics-based loss component.
- `optimizer = OptimizationOptimisers.Adam`: The optimization algorithm to use.
- `learning_rate::Float64 = 0.001`: The learning rate for the optimizer.
- `reltol::Float64 = 1e-6`: The relative tolerance for the ODE solver.
- `abstol::Float64 = 1e-6`: The absolute tolerance for the ODE solver.
- `dtmax = Inf`: The maximum time step for the ODE solver.
- `iters::Int = 1000`: The number of training iterations.
- `rng::StableRNG` = StableRNG(5958): A random number generator for reproducibility.
- `loss_logfile::String = "training_logs/loss_history.txt"`: File path to log loss history.
- `data_vars::Union{Nothing,Vector{Int}} = nothing`: Indices of variables to include in data loss.
- `physics_vars::Union{Nothing,Vector{Int}} = nothing`: Indices of variables to include in physics loss.

# Returns
- `Tuple{Any, Any}`: The trained parameters of the neural network.
"""

function PINN_Infuser_new(
    ode_problem::SciMLBase.ODEProblem,
    ode_params,
    nn::Lux.Chain,
    training_steps::AbstractRange,
    target_data::AbstractMatrix{Float64};
    early_stopping::Bool = true,
    nn_output_weight::Float64 = 0.1,
    physics_weight::Float64 = 1.0,
    optimizer = ADAM,
    learning_rate::Float64 = 1e-3,
    reltol::Float64 = 1e-6,
    abstol::Float64 = 1e-6,
    dtmax = Inf,
    iters::Int = 1000,
    rng::StableRNG = StableRNG(5958),
    loss_logfile::String = "training_logs/loss_history.txt",
    nn_vars::Union{Nothing,Vector{Int}} = nothing,
    data_vars::Union{Nothing,Vector{Int}} = nothing,
    physics_vars::Union{Nothing,Vector{Int}} = nothing,
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
            du[i] *= 1 .+ nn_output_weight .* tanh.(nn_output[k])
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
    
    function plot_solution(temp_sol; target_data = nothing, training_steps = nothing, variable_names = nothing,)
        t = temp_sol.t
        sol_mat = hcat(temp_sol.u...)'   
        n_vars = size(sol_mat, 2)
        plots = [
            begin
                p = plot(t, sol_mat[:, i], label = "2 CHAMBER", xlabel = "time", ylabel = variable_names[i], lw = 2)
                plot!(t, target_data[:, i], label = "TARGET 4 CHAMBER", xlabel = "time", ylabel = variable_names[i], lw = 2)
                p
            end
            for i in 1:10
        ]
        plot(
            plots...,
            layout = (5, 2),
            size = (900, 800)
        )
    end

    function compute_nn_contributions(sol, p_NN)
        t = sol.t
        u_mat = hcat(sol.u...)'  # (time × variables)
        nn_contrib = zeros(length(nn_vars))
        for (k, i) in enumerate(nn_vars)
            nn_outputs_over_time = [
                nn(u_mat[ti, :], p_NN, st)[1][k] for ti in 1:length(t)
            ]
            nn_contrib[i] = nn_output_weight * sum(nn_outputs_over_time)
        end
        return nn_contrib
    end

    function data_loss(pred_norm, data_norm, data_vars)
        return sum(mean(abs2, pred_norm[:, j] .- data_norm[:, j]) for j in data_vars)
    end

    function physics_loss(pred, p_NN, physics_vars)
        pred_mat = hcat(pred.u...)'
        l = 0.0
        for (i, t) in enumerate(training_steps)
            u = pred_mat[i, :]
            du = similar(u)
            pinn_ode!(du, u, p_NN, t)
            f_base = similar(u)
            ode_problem.f(f_base, u, ode_params, t)
            l += mean(abs2.(du[physics_vars] .- f_base[physics_vars]))
        end
        return l / length(training_steps)
    end

    function loss(p_NN)
        pred = predict(p_NN)
        pred_mat = hcat(pred.u...)'
        pred_norm = (pred_mat .- U_MEAN') ./ U_STD'
        L_data = data_loss(pred_norm, data_norm, data_vars)
        L_phy = physics_loss(pred, p_NN, physics_vars)        
        return 1e-2 * L_data + 1e-4 * L_phy
    end

    adtype = Optimization.AutoForwardDiff()
    optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, p_NN)
    losses = Float64[]
    nn_history = Vector{Vector{Float64}}()

    callback = function (state, l)
        pred = predict(state.u)
        pred_mat = hcat(pred.u...)'
        pred_norm = (pred_mat .- U_MEAN') ./ U_STD'
        l_data = data_loss(pred_norm, data_norm, data_vars)
        l_phy = physics_loss(pred, state.u, physics_vars)

        nn_contrib = compute_nn_contributions(pred, state.u)
        push!(losses, 1e-2 * l_data + l_phy)
            push!(nn_history, nn_contrib) 
            println(
            "Iter $(length(losses)) | " *
            "Total: $(round(1e-2 * l_data + 1e-4 * l_phy, sigdigits=5)) | " *
            "Data: $(round(l_data, sigdigits=5)) | " *
            "Physics: $(round(l_phy, sigdigits=5)) | " 
        )
        if length(losses) % 5 == 0
            plt = plot_solution(
                pred;
                target_data = target_data,
                variable_names = ["pLV", "pLA", "psa", "psv", "Vlv", "vLA", "Qav", "Qmv", "Qs", "Qsv"]
            )
            display(plt)
        end
        if early_stopping &&
            length(losses) > 50 &&
            losses[end] - maximum(losses[(end-10):(end-1)]) > 0
            println("Early stopping at iteration $(length(losses)) with loss $(losses[end])")
            return true
        else
            return false
        end
    end

    trained_params = Optimization.solve(
        optprob,
        optimizer(learning_rate),
        callback = callback,
        maxiters = iters,
    )

    folder = dirname(loss_logfile)
    if folder != "" && !isdir(folder)
        println("Creating directory for training logs: $folder")
        mkpath(folder)
    end
    open(loss_logfile, "w") do io
        for (i, L) in enumerate(losses)
            @printf(io, "%d %.12f\n", i, L)
        end
    end
    return (trained_params.u, st, losses, nn_history)
end

end # module PINNInfuser