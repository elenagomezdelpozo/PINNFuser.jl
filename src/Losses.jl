module Losses

using StableRNGs, ComponentArrays, LinearAlgebra
using Optimization, OptimizationOptimisers
using SciMLBase, SciMLSensitivity
using Statistics, ForwardDiff
using OrdinaryDiffEq: Vern7, Tsit5, Rodas4
using Printf
using Plots, Zygote, Statistics

export Loss

# ─── individual loss calculators ──────────────────────────────────────────────

function data_loss(pred_norm, data_norm, data_vars, data_weight)
    return data_weight * sum(mean(abs2, pred_norm[:, j] .- data_norm[:, j]) for j in data_vars)
end

function physics_loss(pred_mat, training_steps, ode_params, physics_vars, physics_weight, ode_problem)
    dt = training_steps[2] - training_steps[1]
    n  = size(pred_mat, 1)
    l  = zero(eltype(pred_mat))
    du_scale = vec(std(diff(pred_mat, dims=1) ./ dt, dims=1)) .+ 1e-6

    for i in 2:n-1
        t = training_steps[i]
        u = pred_mat[i, :]
        du_ode = zeros(eltype(u), length(u))
        ode_problem.f(du_ode, u, ode_params, t)
        du_num = (pred_mat[i+1, :] .- pred_mat[i-1, :]) ./ (2dt)
        for j in physics_vars
            l += abs2((du_num[j] - du_ode[j]) / du_scale[j])
        end
    end

    return physics_weight * l / (n - 2)
end

function mass_conservation_loss(pred_mat, training_steps, mass_conservation_weight)
    dt = training_steps[2] - training_steps[1]
    Qav, Qmv, Qs, Qsv = pred_mat[:, 7], pred_mat[:, 8], pred_mat[:, 9], pred_mat[:, 10]

    ∫Qav  = sum((Qav[1:end-1]  .+ Qav[2:end])  ./ 2) * dt
    ∫Qmv  = sum((Qmv[1:end-1]  .+ Qmv[2:end])  ./ 2) * dt
    ∫Qs   = sum((Qs[1:end-1]   .+ Qs[2:end])   ./ 2) * dt
    ∫Qsv  = sum((Qsv[1:end-1]  .+ Qsv[2:end])  ./ 2) * dt

    mean_flow = (∫Qav + ∫Qmv + ∫Qs + ∫Qsv) / 4
    l = abs2(∫Qav - mean_flow) + abs2(∫Qmv - mean_flow) +
        abs2(∫Qs  - mean_flow) + abs2(∫Qsv - mean_flow)

    return mass_conservation_weight * l
end

function zero_mean_loss(sol, p_NN, nn, st, nn_vars, zm_vars, zm_weight)
    t     = sol.t
    u_mat = hcat(sol.u...)'
    l     = 0.0
    for (k, i) in enumerate(nn_vars)
        if i in zm_vars
            nn_over_time = [nn(u_mat[ti, :], p_NN, st)[1][k] for ti in 1:length(t)]
            l += abs2(sum(nn_over_time))
        end
    end
    return zm_weight * l
end

function negativity_loss(pred_mat, neg_vars, neg_weight)
    l = 0.0
    for j in neg_vars
        l += abs2(sum(min.(pred_mat[:, j], 0.0)))
    end
    return neg_weight * l
end

function firstderiv_loss(pred_mat, target_data, firstderiv_vars, training_steps, deriv_weight)
    dt = training_steps[2] - training_steps[1]
    l  = 0.0
    for j in firstderiv_vars
        dpred   = diff(pred_mat[:, j],    dims=1) ./ dt
        dtarget = diff(target_data[:, j], dims=1) ./ dt
        l += mean(abs2, dpred .- dtarget)
    end
    return deriv_weight * l
end

function periodicity_loss(pred_mat, periodic_weight)
    l = 0.0
    for j in 1:10
        l += abs2(pred_mat[1, j] - pred_mat[end, j])
    end
    return periodic_weight * l
end

function loss(active::Vector{String}, ctx, config)

    REGISTRY = Dict{String, Function}(
        "data"        => () -> data_loss(ctx.pred_norm, ctx.data_norm,
                                         config.data_vars, config.data_weight),
        "physics"     => () -> physics_loss(ctx.pred_mat, ctx.training_steps,
                                            ctx.ode_params, config.physics_vars,
                                            config.physics_weight, ctx.ode_problem),
        "mass"        => () -> mass_conservation_loss(ctx.pred_mat,
                                                      ctx.training_steps,
                                                      config.mass_conservation_weight),
        "zero_mean"   => () -> zero_mean_loss(ctx.sol, ctx.p_NN, ctx.nn, ctx.st,
                                              ctx.nn_vars, config.zm_vars, config.zm_weight),
        "negativity"  => () -> negativity_loss(ctx.pred_mat, config.neg_vars,
                                               config.neg_weight),
        "firstderiv"  => () -> firstderiv_loss(ctx.pred_mat, ctx.target_data,
                                               config.firstderiv_vars,
                                               ctx.training_steps, config.deriv_weight),
        "periodicity" => () -> periodicity_loss(ctx.pred_mat, config.periodic_weight),
    )

    # validate requested keys
    unknown = setdiff(active, keys(REGISTRY))
    isempty(unknown) || @warn "Unknown loss keys ignored: $unknown"

    # evaluate only the requested losses
    results = Dict{Symbol, Float64}()
    total   = 0.0

    for key in active
        haskey(REGISTRY, key) || continue
        val            = REGISTRY[key]()
        results[Symbol(key)] = val
        total         += val
    end

    results[:total] = total
    return NamedTuple(results)
end # function

end # module