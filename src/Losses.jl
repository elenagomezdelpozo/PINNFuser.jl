__precompile__(false)  # Add this here
module LossesMod

using ComponentArrays, LinearAlgebra
using SciMLBase
using Statistics
using Zygote  

export loss

function data_loss(pred_norm, data_norm, data_vars, data_weight)
    return data_weight * sum(mean(abs2, pred_norm[:, j] .- data_norm[:, j]) for j in data_vars)
end

function physics_loss(pred_mat, training_steps, ode_params, physics_vars, physics_weight, ode_problem)
    dt = training_steps[2] - training_steps[1]
    n  = size(pred_mat, 1)
    l  = zero(eltype(pred_mat))
    du_scale = Zygote.ignore() do
        vec(std(diff(Array(pred_mat), dims=1) ./ dt, dims=1)) .+ 1e-6
    end
    for i in 2:n-1
        t  = training_steps[i]
        u  = pred_mat[i, :]
        du_ode = Zygote.ignore() do
            Vector{Float64}(ode_problem.f(Vector{Float64}(u), ode_params, t))
        end
        du_num = (pred_mat[i+1, :] .- pred_mat[i-1, :]) ./ (2 * dt)
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

function zero_mean_loss(u_mat, p_NN, nn, st, nn_vars, zm_vars, zm_weight)
    l = zero(eltype(u_mat))
    for (k, i) in enumerate(nn_vars)
        if i in zm_vars
            nn_over_time = [nn(u_mat[ti, :], p_NN, st)[1][k] for ti in 1:size(u_mat, 1)]
            l += abs2(sum(nn_over_time))
        end
    end
    return zm_weight * l
end

function negativity_loss(pred_mat, neg_vars, neg_weight)
    l = zero(eltype(pred_mat)) 
    for j in neg_vars
        l += abs2(sum(min.(pred_mat[:, j], 0.0)))
    end
    return neg_weight * l
end

function firstderiv_loss(pred_mat, target_data, firstderiv_vars, training_steps, deriv_weight)
    dt = training_steps[2] - training_steps[1]
    l  = zero(eltype(pred_mat))
    for j in firstderiv_vars
        dpred   = diff(pred_mat[:, j],    dims=1) ./ dt
        dtarget = diff(target_data[:, j], dims=1) ./ dt
        l += mean(abs2, dpred .- dtarget)
    end
    return deriv_weight * l
end

function periodicity_loss(pred_mat, periodic_weight, periodic_vars)
    l  = zero(eltype(pred_mat))
    for j in periodic_vars
        l += abs2(pred_mat[1, j] - pred_mat[end, j])
    end
    return periodic_weight * l
end

function loss(active::Vector{String}, ctx, config)
    total = zero(eltype(ctx.pred_mat))
    for key in active
        val = if key == "data"   
                data_loss(ctx.pred_norm, ctx.data_norm, 
                            config.data_vars, config.data_weight)   
                elseif key == "physics"
                    physics_loss(ctx.pred_mat, ctx.training_steps,
                                ctx.ode_params, config.physics_vars,
                                config.physics_weight, ctx.ode_problem)
                elseif key == "mass"
                    mass_conservation_loss(ctx.pred_mat, ctx.training_steps,
                                        config.mass_conservation_weight)
                elseif key == "zero_mean"
                    zero_mean_loss(ctx.pred_mat, ctx.p_NN, ctx.nn, ctx.st,
                                ctx.nn_vars, config.zm_vars, config.zm_weight)
                elseif key == "negativity"
                    negativity_loss(ctx.pred_mat, config.neg_vars, config.neg_weight)
                elseif key == "firstderiv"
                    firstderiv_loss(ctx.pred_mat, ctx.target_data,
                                    config.firstderiv_vars,
                                    ctx.training_steps, config.deriv_weight)
                elseif key == "periodicity"
                    periodicity_loss(ctx.pred_mat, config.periodic_weight,
                                    config.periodic_vars)
        else; nothing; end
        isnothing(val) || (total += val)
    end
    return (total = total,)   # only field Zygote needs to trace
end

end # module