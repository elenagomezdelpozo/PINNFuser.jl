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
        ode_problem.f(du, u, nothing, t)
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
    function plot_solution(temp_sol; target_data = nothing, training_steps = nothing, variable_names = nothing,)
        t = temp_sol.t
        sol_mat = hcat(temp_sol.u...)'   # (time × variables)
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
            nn_contrib[i] = nn_output_weight * mean(nn_outputs_over_time)
        end
        return nn_contrib
    end

    function data_1deriv_loss(p_NN, pred_sol, target_data, training_steps, data_vars)
        dt = step(training_steps)
        n_steps = size(target_data, 1)
        l = 0.0

        d1_target = zeros(size(target_data))
        for i in 2:n_steps-1
            d1_target[i, :] .= (target_data[i+1, :] .- target_data[i-1, :]) ./ (2 * dt)
        end

        d1_target[1,:] .= (target_data[2, :] .- target_data[1, :]) ./ dt
        d1_target[end, :] .= (target_data[end, :] .- target_data[end-1, :]) ./ dt
        du_pinn = zeros(eltype(p_NN), size(target_data, 2))
        for ti in 1:n_steps
            u_true = target_data[ti, :]
            t_now = training_steps[ti]
            pinn_ode!(du_pinn, u_true, p_NN, t_now)
            for j in data_vars
                diff = (du_pinn[j] - d1_target[ti, j]) / U_STD[j]
                l += abs2(diff)
            end
        end

        return l / (n_steps * length(data_vars))
    end

    function data_loss(pred_norm, data_norm, data_vars)
        return sum(mean(abs2, pred_norm[:, j] .- data_norm[:, j]) for j in data_vars)
    end

    function physics_loss(pred_mat, p_NN)
        l_phy = 0.0
        for (i, t) in enumerate(training_steps)
            u = pred_mat[i, :]
            du = similar(u)
            pinn_ode!(du, u, p_NN, t) # derivatives with increments from PINN
            du_base = similar(du)
            ode_problem.f(du_base, u, nothing, t) # derivatives from base ODE
            l_phy += mean(abs2.(du[physics_vars] .- du_base[physics_vars]))
        end
        return l_phy / length(training_steps)
    end

    function loss(p_NN)
        pred = predict(p_NN)
        pred_mat = hcat(pred.u...)'
        pred_norm = (pred_mat .- U_MEAN') ./ U_STD'
        L_1deriv = data_1deriv_loss(p_NN, pred, target_data, training_steps, data_vars)
        L_data = data_loss(pred_norm, data_norm, data_vars)
        # L_phy = physics_loss(pred_mat, p_NN)
        return 1e-2 * L_data + 1e-3 *L_1deriv
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
        l_1deriv = data_1deriv_loss(state.u, pred, target_data, training_steps, data_vars)
        l_data = data_loss(pred_norm, data_norm, data_vars)
        # l_phy = physics_loss(pred_mat, state.u)

        nn_contrib = compute_nn_contributions(pred, state.u)
        push!(losses, 1e-2 * l_data + 1e-3 * l_1deriv)
        push!(nn_history, nn_contrib) 
        println(
            "Iter $(length(losses)) | " *
            "Total: $(round(l_data + l_1deriv, sigdigits=5)) | " *
            # "First derivative: $(round(l_1deriv, sigdigits=5)) | " *
            "Data: $(round(l_data, sigdigits=5)) | " *
            "1st Deriv: $(round(l_1deriv, sigdigits=5)) | " 
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