__precompile__(false)  # Add this here
module LossesMod

using ComponentArrays, LinearAlgebra
using SciMLBase, Statistics, Zygote  

export loss

# Now data loss only sees the maximum and minimum values of the predicted and target data, not the full matrices.
function data_loss(pred_norm, data_norm, data_vars, data_weight)
    l = zero(eltype(pred_norm))
    for j in data_vars
        l += abs2(maximum(pred_norm[:, j]) - maximum(data_norm[:, j]))
        l += abs2(minimum(pred_norm[:, j]) - minimum(data_norm[:, j]))
    end
    return data_weight * l / length(data_vars)
end

function physics_loss(pred_mat, training_steps, ode_params, physics_vars, physics_weight, ode_problem, nn, p_NN, st, dt, nn_vars=nothing)
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

        # Only compute NN contribution if nn_vars is defined and not empty
        du_num = (pred_mat[i+1, :] .- pred_mat[i-1, :]) ./ (2 * dt)
        
        if !isnothing(nn_vars) && length(nn_vars) > 0
            nn_contrib_full = nn(u, p_NN, st)[1]  # NN correction, must stay in AD graph
            for j in physics_vars
                # Find the index in nn_vars if j is in nn_vars, otherwise use ODE only
                idx = findfirst(==(j), nn_vars)
                correction = isnothing(idx) ? zero(eltype(pred_mat)) : nn_contrib_full[idx]
                residual = du_num[j] - (du_ode[j] + correction)
                l += abs2(residual / du_scale[j])
            end
        else
            # Use ODE only, no NN correction
            for j in physics_vars
                residual = du_num[j] - du_ode[j]
                l += abs2(residual / du_scale[j])
            end
        end
    end

    return physics_weight * l / (n - 2)
end

function mass_conservation_loss(pred_mat, mass_conservation_weight, dt)
    Qav, Qmv, Qs, Qsv = pred_mat[:, 7], pred_mat[:, 8], pred_mat[:, 9], pred_mat[:, 10]

    ∫Qav  = sum((Qav[1:end-1]  .+ Qav[2:end])  ./ 2) * dt
    ∫Qmv  = sum((Qmv[1:end-1]  .+ Qmv[2:end])  ./ 2) * dt
    ∫Qs   = sum((Qs[1:end-1]   .+ Qs[2:end])   ./ 2) * dt
    ∫Qsv  = sum((Qsv[1:end-1]  .+ Qsv[2:end])  ./ 2) * dt

    mean_flows = (∫Qav , ∫Qmv , ∫Qs , ∫Qsv) 
    l = zero(eltype(pred_mat))
    # Comparing all pairs of mean flows: (av-mv, av-s, av-sv, mv-s, mv-sv, s-sv)
    for i in 1:3
        for j in i+1:4
            l += abs2(mean_flows[i] - mean_flows[j])
        end
    end
    return mass_conservation_weight * l / 6 # 6 pairs
end

function zero_mean_loss(pred_mat, p_NN, nn, st, nn_vars, zm_vars, zm_weight)
    l = 0.0
    time = size(pred_mat, 1)
    count = 0
    for (k, i) in enumerate(nn_vars)
        if i in zm_vars
            # nn_over_time collects the k-th output of the NN (corresponding to state i) over time
            nn_over_time = [nn(pred_mat[ti, :], p_NN, st)[1][k] for ti in 1:time]
            l += abs2(sum(nn_over_time))
            count += 1
        end
    end
    return count > 0 ? zm_weight * l / count : zero(l)
end

function negativity_loss(pred_mat, neg_vars, neg_weight)
    l = zero(eltype(pred_mat)) 
    for j in neg_vars
        l += mean(abs2, min.(pred_mat[:, j], zero(eltype(pred_mat))))
    end
    return neg_weight * l
end

function periodicity_loss(pred_mat, periodic_weight, periodic_vars)
    l  = zero(eltype(pred_mat))
    for j in periodic_vars
        l += abs2(pred_mat[1, j] - pred_mat[end, j])
    end
    return periodic_weight * l / length(periodic_vars)
end

function firstderiv_loss(pred_mat, target_data, firstderiv_vars, deriv_weight, dt)
    # Compute first derivatives of predictions and targets
    n = size(pred_mat, 1)
    l = zero(eltype(pred_mat))
    
    # Compute numerical derivatives using central difference
    pred_deriv = similar(pred_mat)
    target_deriv = similar(target_data)
    
    # Forward difference at t=1
    pred_deriv[1, :] .= (pred_mat[2, :] .- pred_mat[1, :]) ./ dt
    target_deriv[1, :] .= (target_data[2, :] .- target_data[1, :]) ./ dt
    
    # Central differences for internal points
    for i in 2:n-1
        pred_deriv[i, :] .= (pred_mat[i+1, :] .- pred_mat[i-1, :]) ./ (2 * dt)
        target_deriv[i, :] .= (target_data[i+1, :] .- target_data[i-1, :]) ./ (2 * dt)
    end
    
    # Backward difference at t=n
    pred_deriv[n, :] .= (pred_mat[n, :] .- pred_mat[n-1, :]) ./ dt
    target_deriv[n, :] .= (target_data[n, :] .- target_data[n-1, :]) ./ dt
    
    # Compute loss on specified variables
    for j in firstderiv_vars
        l += mean(abs2, pred_deriv[:, j] .- target_deriv[:, j])
    end
    
    return deriv_weight * l / length(firstderiv_vars)
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
                                config.physics_weight, ctx.ode_problem,
                                ctx.nn, ctx.p_NN, ctx.st, config.dt, ctx.nn_vars)
                elseif key == "mass"
                    mass_conservation_loss(ctx.pred_mat,
                                        config.mass_conservation_weight, config.dt)
                elseif key == "zero_mean"
                    zero_mean_loss(ctx.pred_mat, ctx.p_NN, ctx.nn, ctx.st,
                                ctx.nn_vars, config.zm_vars, config.zm_weight)
                elseif key == "negativity"
                    negativity_loss(ctx.pred_mat, config.neg_vars, config.neg_weight)
                elseif key == "firstderiv"
                    firstderiv_loss(ctx.pred_mat, ctx.target_data,
                                    config.firstderiv_vars, config.deriv_weight, 
                                    config.dt)
                elseif key == "periodicity"
                    periodicity_loss(ctx.pred_mat, config.periodic_weight,
                                    config.periodic_vars)
        else; nothing; end
        isnothing(val) || (total += val)
    end
    return (total = total,)   # only field Zygote needs to trace
end

end # module