module PINNInfuser_new

__precompile__(false)   # fixes the rrule/ChainRules conflict

using Lux
using StableRNGs, ComponentArrays, LinearAlgebra
using Optimization, OptimizationOptimisers
using SciMLBase, SciMLSensitivity
using Statistics, ForwardDiff
using OrdinaryDiffEq: Vern7, Tsit5, Rodas4
using Printf
using Plots, Zygote, Statistics

include("Losses.jl")
using .Losses

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
    active::Vector{String},
    config::NamedTuple,
    nn_output_weight::Float64 = 1.0,
    optimizer = ADAM,
    learning_rate::Float64 = 1e-3,
    reltol::Float64 = 1e-6,
    abstol::Float64 = 1e-6,
    dtmax = Inf,
    iters::Int = 1000,
    early_stopping::Bool = true,
    plot_every::Int = 10,
    early_stop_warmup::Int = 50,
    early_stop_window::Int = 10,
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
            (du, u, p, t) -> cpu_pinn_ode!(du, u, p, t), 
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

    function compute_nn_contributions(sol, p_NN)
        t = sol.t
        u_mat = hcat(sol.u...)'  # (time × variables)
        # Return (time × nn_vars) matrix instead of summing
        nn_contrib = zeros(length(t), length(nn_vars))
        for (k, i) in enumerate(nn_vars)
            for ti in 1:length(t)
                nn_contrib[ti, k] = nn_output_weight * nn(u_mat[ti, :], p_NN, st)[1][k]
            end
        end
        return nn_contrib  # (time × nn_vars)
    end

    ctx = (
        pred_mat    = pred_mat,
        pred_norm   = pred_norm,
        data_norm   = data_norm,
        training_steps = training_steps,
        ode_params  = ode_params,
        ode_problem = ode_problem,
        nn_vars = nn_vars
        )

    adtype = Optimization.AutoForwardDiff()
    optf = Optimization.OptimizationFunction((x, p) -> loss(active, ctx, config), adtype)
    optprob = Optimization.OptimizationProblem(optf, p_NN)
    losses = Float64[]
    nn_history = Vector{Matrix{Float64}}()

    callback = function (state, l)
        pred = cpu_predict(state.u)
        loss = loss(active, ctx, config)
        push!(losses, loss)
        push!(nn_history, cpu_compute_nn_contributions(pred, state.u))
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

    return (trained_params.u, st, losses, nn_history) 
end # function
end # module