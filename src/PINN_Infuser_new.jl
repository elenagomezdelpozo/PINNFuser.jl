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
    params,
    training_steps::AbstractRange,
    target_data::AbstractMatrix{Float64};
    early_stopping::Bool = true,
    nn_output_weight::Float64 = 0.1,
    physics_weight::Float64 = 1.0,
    optimizer = ADAM,
    learning_rate::Float64 = 0.001,
    reltol::Float64 = 1e-6,
    abstol::Float64 = 1e-6,
    dtmax = Inf,
    iters::Int = 1000,
    rng::StableRNG = StableRNG(5958),
    loss_logfile::String = "training_logs/loss_history.txt",
    nn_vars::Union{Nothing,Vector{Int}} = nothing,
    data_vars::Union{Nothing,Vector{Int}} = nothing,
    physics_vars::Union{Nothing,Vector{Int}} = nothing,
)::Tuple{Any,Any,Vector{Float64},Vector{Float64}}

    U_MEAN = vec(mean(target_data, dims=1))
    U_STD  = vec(std(target_data, dims=1)) .+ 1e-6
    target_data_norm = (target_data .- U_MEAN') ./ U_STD'
    
    if nn_vars === nothing
        nn_vars = collect(1:length(ode_problem.u0))
    end    
    data_vars === nothing && (data_vars = nn_vars)
    physics_vars === nothing && (physics_vars = nn_vars)

    p_NN, st = Lux.setup(rng, nn)
    p_NN = 1e-3 * ComponentVector{Float64}(p_NN) # NN parameter initialization

    function pinn_ode!(du, u, p_NN, t)
        nn_output = nn(u, p_NN, st)[1]
        ode_problem.f(du, u, params, t)
        for i in nn_vars
            du[i] += nn_output_weight * nn_output[i]
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
            abstol=abstol)
        return temp_sol

    end

    # Loss data
    function data_loss(pred_norm, data_norm, data_vars)
        return sum(mean(abs2, pred_norm[:, j] .- data_norm[:, j]) for j in data_vars)
    end

    function data_deriv_loss(pred_norm, data_norm, data_vars)
        l = 0
        for j in data_vars
            d_pred = diff(pred_norm[:, j])
            d_data = diff(data_norm[:, j])
            l += mean(abs2, d_pred .- d_data)
        end
        return l * 100 # derivative loss weight
    end
    
    function negativity_loss(pred_norm)
        pred_sub = pred_norm[:, 4:6] # Vlv, Qav, Qmv
        l = sum(relu.(-pred_sub).^2)
        return 10 * l / length(pred_sub)
    end

    function periodic_loss(pred_norm)
        start_u = pred_norm[1, :]
        end_u   = pred_norm[end, :]
        return mean(abs2.(end_u - start_u))
    end

    """
    # Loss physics
    function physics_loss(pred_norm, p_NN)
        l_phy = 0.0
        for (i, t) in enumerate(training_steps)
            u = pred_mat[i, :]
            du = similar(u)
            pinn_ode!(du, u, p_NN, t, params) # derivatives with increments from PINN
            du_base = similar(du)
            ode_problem.f(du_base, u, params, t) # derivatives from base ODE
            l_phy += mean(abs2.(du[physics_vars] .- du_base[physics_vars]))
        end
        return l_phy / length(training_steps)
    end
    """
    function loss(p_NN)
        pred = predict(p_NN)
        pred_mat = hcat(pred.u...)'
        pred_norm = (pred_mat .- U_MEAN') ./ U_STD'
        l_data = data_loss(pred_norm, target_data_norm, data_vars)
        l_data_der = data_deriv_loss(pred_norm, target_data_norm, data_vars)
        l_neg = negativity_loss(pred_norm) # not giving norm because it diminishes the loss
        l_periodic = periodic_loss(pred_norm)
        return l_data + l_data_der + l_neg + l_periodic
    end

    # Optimization
    adtype = Optimization.AutoForwardDiff()
    optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, p_NN)
    losses = Float64[]

    callback = function(p_NN, l)
        pred = predict(p_NN.u)
        pred_mat = hcat(pred.u...)'
        pred_norm = (pred_mat .- U_MEAN') ./ U_STD'

        # Individual losses
        l_data = data_loss(pred_norm, target_data_norm, data_vars)
        l_data_der = data_deriv_loss(pred_norm, target_data_norm, data_vars)
        l_neg = negativity_loss(pred_norm)
        l_periodic = periodic_loss(pred_norm)

        push!(losses, l)

        println(
            "Iter $(length(losses)) | " *
            "Total: $(round(l, sigdigits=5)) | " *
            "Data: $(round(l_data, sigdigits=5)) | " *
            "Deriv: $(round(l_data_der, sigdigits=5)) | " *
            "Neg: $(round(l_neg, sigdigits=5)) | " *
            "Periodic: $(round(l_periodic, sigdigits=5))"
        )

        if early_stopping && length(losses) > 50 &&
        losses[end] > maximum(losses[end-10:end-1])
            println("Early stopping at iter $(length(losses))")
            return true
        end

        return false
    end


    trained_params = Optimization.solve(
        optprob,
        optimizer(learning_rate),
        callback=callback,
        maxiters=iters,
    )

    # Guardar historial de pérdidas
    folder = dirname(loss_logfile)
    if folder != "" && !isdir(folder)
        mkpath(folder)
    end
    open(loss_logfile, "w") do io
        for (i,L) in enumerate(losses)
            @printf(io, "%d %.12f\n", i, L)
        end
    end
    return (trained_params.u, st, U_MEAN, U_STD)
end

end # module