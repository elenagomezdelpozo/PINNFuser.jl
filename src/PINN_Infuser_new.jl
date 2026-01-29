module PINNInfuser_new

using Lux, StableRNGs, Optimization, OptimizationOptimisers, ComponentArrays, LinearAlgebra
using OrdinaryDiffEq, Statistics, ForwardDiff
using OrdinaryDiffEq: Tsit5, Rodas4
using Printf
using Plots
using Lux, Plots, Zygote, Statistics, StableRNGs, ComponentArrays

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
    period_weight::Float64 = 1.0,
    slope_weight::Float64 = 1e-6,
    optimizer = ADAM,
    learning_rate::Float64 = 0.001,
    reltol::Float64 = 1e-6,
    abstol::Float64 = 1e-6,
    dtmax = Inf,
    iters::Int = 1000,
    rng::StableRNG = StableRNG(5958),
    loss_logfile::String = "training_logs/loss_history.txt",
    nn_vars::Union{Nothing,Vector{Int}} = nothing,
    phy_vars::Union{Nothing,Vector{Int}} = nothing,
    smooth_vars::Union{Nothing,Vector{Int}} = nothing,
    τ::Float64 = 1.0,
)::Tuple{Any,Any,Any,Any}
    
    nn_vars === nothing && (nn_vars = collect(1:length(ode_problem.u0)))
    phy_vars === nothing && (phy_vars = collect(1:length(ode_problem.u0)))
    smooth_vars === nothing && (smooth_vars = collect(1:length(ode_problem.u0)))

    p_NN, st = Lux.setup(rng, nn)
    p_NN = ComponentVector{Float64}(p_NN)

    T = eltype(p_NN) 
    U_MEAN = T.(reshape(mean(target_data, dims = 1), 1, :))
    U_STD  = T.(reshape(std(target_data, dims = 1), 1, :)) .+ T(1e-6)
    target_data_norm = (target_data .- U_MEAN) ./ U_STD

    ode_f = ode_problem.f
    # Slopes of target data for slope loss
    target_slopes = zeros(size(target_data))
    target_slopes = zeros(size(target_data))
    dt_approx = training_steps[2] - training_steps[1]
    for j in 1:size(target_data_norm, 2)
        # Simple central difference
        target_slopes[2:end-1, j] = (target_data_norm[3:end, j] .- target_data_norm[1:end-2, j]) ./ (2*dt_approx)
        target_slopes[1, j] = target_slopes[2, j] # boundary fill
        target_slopes[end, j] = target_slopes[end-1, j]
    end
    S_MEAN = vec(mean(target_slopes, dims=1))
    S_STD = vec(std(target_slopes, dims=1)) .+ 1e-6
    target_slopes_norm = (target_slopes .- S_MEAN') ./ S_STD'

    function pinn_ode!(du, u, p_weights, t) # Current weights
        ode_f(du, u, params, t) 
        
        φ = mod(t, τ) / τ
        nn_input = vcat((u .- vec(U_MEAN)) ./ vec(U_STD), T(sin(2π*φ)), T(cos(2π*φ)))
        nn_correction = nn(nn_input, p_weights, st)[1]

        for (i_local, i_global) in enumerate(nn_vars)
            du[i_global] += nn_output_weight * tanh(nn_correction[i_local])
        end    
    end

    function predict(p_weights)
        new_prob = ODEProblem(
            (du, u, p, t) -> pinn_ode!(du, u, p, t),
            ode_problem.u0,
            ode_problem.tspan,
            p_weights
        )

        return solve(
            new_prob,
            Vern7(),
            saveat = training_steps,
            dtmax = dtmax,
            reltol = reltol,
            abstol = abstol,
        )
        if sol.retcode != ReturnCode.Success || length(sol.u) != length(training_steps)
            return fill(NaN, length(training_steps), length(ode_problem.u0))
        end
        return hcat(sol.u...)'
    end
    function data_loss(pred_norm, data, nn_vars)
        return sum(mean(abs2, pred_norm[:, j] .- data[:, j]) for j in nn_vars) 
    end

    function physics_loss(pred_mat, p_NN, phy_vars)
        l = 0.0
        # Pre-allocate to avoid repeated allocations
        du = similar(pred_mat[1, :])
        f_base = similar(du)
        
        for (i, t) in enumerate(training_steps)
            u = pred_mat[i, :]
            pinn_ode!(du, u, p_NN, t)
            ode_f(f_base, u, params, t) # Fixed: changed 'p' to 'params'
            l += mean(abs2, du[phy_vars] .- f_base[phy_vars])
        end
        return l / length(training_steps) # Normalized by time steps
    end
    
    function periodicity_loss(pred_mat, p_weights)
        # Enforce u(t_end) == u(t_start)
        u_start = pred_mat[1, :]
        u_end = pred_mat[end, :]

        du_start = similar(u_start)
        du_end = similar(u_end)

        pinn_ode!(du_start, u_start, p_weights, training_steps[1])
        pinn_ode!(du_end, u_end, p_weights, training_steps[end])

        # Penalize difference in state AND difference in derivative
        return mean(abs2, u_start .- u_end) + mean(abs2, du_start .- du_end)    
    end
    """
    function slope_loss(pred, p_NN, smooth_vars)
        pred_mat = hcat(pred.u...)'        
        l = 0.0
        for (i, t) in enumerate(training_steps)
            u = pred_mat[i, :]
            du_pred = similar(u)
            pinn_ode!(du_pred, u, p_NN, t)
            l += mean(abs2.(
                du_pred[smooth_vars] .- target_slopes_norm[i, smooth_vars]
            ))
        end
        return l
    end
    """
    function loss(p_weights)
        pred_mat = predict(p_weights)
        pred_mat = hcat(pred_mat.u...)'
        if any(isnan, pred_mat) return 1e6 end # Penalty for crashing
        pred_norm = (pred_mat .- U_MEAN) ./ U_STD
        
        L_data = data_loss(pred_norm, target_data_norm, nn_vars)
        L_phys = physics_loss(pred_mat, p_weights, phy_vars)
        L_per = periodicity_loss(pred_mat, p_weights)

        # L_slope = slope_loss(pred_mat, p_weights, smooth_vars)
        return L_data + 
            (physics_weight * L_phys) + 
            (period_weight * L_per) 
            # (slope_weight * L_slope)   
    end

    adtype = Optimization.AutoForwardDiff()
    optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, p_NN)
    losses = Float64[]

    var_titles = ["pLV", "psa", "psv", "Vlv", "Qav", "Qmv", "Qs"]
    callback = function (p_NN, l)
        push!(losses, l)
        println("Iteration $(length(losses)): Loss = $(losses[end])")
        if length(losses) % 10 == 0
            pred_sol = predict(p_NN.u)  # use the current parameters
            pred_mat = hcat(pred_sol.u...)'

            plots_list = []
            for j in 1:7
                plot1 = plot(training_steps, target_data[:, j], 
                         label="Target", color=:black, lw=2, legend=false,
                         title=var_titles[j], titlefontsize=10)
                
                plot!(plot1, training_steps, pred_mat[:, j], 
                      label="Pred", color=:red, linestyle=:dash, lw=2)
                
                push!(plots_list, plot1)
            end
            final_plot = plot(plots_list..., layout=(7, 1), size=(700, 1400), margin=5Plots.mm)
            display(final_plot)
        end
        if early_stopping &&
            length(losses) > 100 &&
            losses[end] - maximum(losses[(end-10):(end-1)]) > 0
            println("Early stopping at iteration $(length(losses)) with loss $(losses[end])")
            return true
        else
            return false
        end
    end

    result = Optimization.solve(
        optprob,
        optimizer(learning_rate),
        callback = callback,
        maxiters = iters,
    )
    print(nn_vars)

    folder = dirname(loss_logfile)
    if folder != "" && !isdir(folder)
        println("Creating directory for training logs: $folder")
        mkpath(folder)
    end
    print(nn_vars)

    open(loss_logfile, "w") do io
        for (i, L) in enumerate(losses)
            @printf(io, "%d %.12f\n", i, L)
        end
    end


    return (result.u, st, U_MEAN, U_STD)
end

end # module PINNInfuser