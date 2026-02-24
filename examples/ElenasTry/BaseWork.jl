using OrdinaryDiffEq
using Lux, Plots, Zygote, Statistics, StableRNGs, ComponentArrays
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using Optim, Measures, BenchmarkTools
using DelimitedFiles, ForwardDiff
using Plots, LinearAlgebra
include("/Applications/Desktop/CODE/Thesis/PINNFuser.jl/examples/ElenasTry/cvmodelelena.jl")
using .elenasimpleModel
import Main.elenasimpleModel: Elastance_v, Elastance_a, DShiElastance_v, DShiElastance_a, Valve


τ = 1.0 # Cardiac cycle period
τₑₛ_lv, τₑₚ_lv, τₑₛ_la, τₑₚ_la = 0.3, 0.45, 0.25, 0.15
Rmv, Zao, Rs, Rsv, Csa, Csv, Eₘₐₓ_lv, Eₘᵢₙ_lv, Eₘₐₓ_la, Eₘᵢₙ_la = p_pred
ode_params = [τₑₛ_lv, τₑₚ_lv, Rmv, Zao, Rs, Rsv, Csa, Csv, Eₘₐₓ_lv, Eₘᵢₙ_lv, τₑₛ_la, τₑₚ_la, Eₘₐₓ_la, Eₘᵢₙ_la]
u0 = [7.0, 7.0, 7.0, 7.0, 200.0, 60.0, 0.0, 0.0, 0.0, 0.0] 
Eshift = 0.0

tspan = (5.0, 6.0)
tsteps = range(5.0, 6.0 , length = 150)
loaded_data = readdlm("/Applications/Desktop/CODE/Data_acquisition/data/original_data_new.txt") # NEW DATA ACQUISITION METHOD
extrap_original_data = Array{Float64}(loaded_data)[1:Int(floor(40 * 150)), :] # 40 seconds of data, 150 samples per second
original_data = extrap_original_data[751:750 + 150, :]
extrap_tseps = range(0, 40, length = 40 * 150)

"""
# Visualize Elastances
t_range = 0:0.001:τ
E_v = [Elastance_v(t, Eₘᵢₙ_lv, Eₘₐₓ_lv, τ, τₑₛ_lv, τₑₚ_lv, Eshift) for t in t_range]
E_a = [Elastance_a(t, Eₘᵢₙ_la, Eₘₐₓ_la, τ, τₑₛ_la, τₑₚ_la) for t in t_range]
plot(t_range, E_v, label="LV Elastance", lw=2, color=:blue, title="Heart Chamber Elastance Timing")
plot!(t_range, E_a, label="LA Elastance", lw=2, color=:red)
xlabel!("Time (s)")
ylabel!("Elastance (mmHg/mL)")
"""

function NIK_2ch!(du, u, p, t)
    pLV, pLA, psa, psv, Vlv, Vla, Qav, Qmv, Qs, Qsv = u
    du[1] =
        (Qmv - Qav) * Elastance_v(t, Eₘᵢₙ_lv, Eₘₐₓ_lv, τ, τₑₛ_lv, τₑₚ_lv, Eshift) +
        (pLV / Elastance_v(t, Eₘᵢₙ_lv, Eₘₐₓ_lv, τ, τₑₛ_lv, τₑₚ_lv, Eshift))*
        DShiElastance_v(t, Eₘᵢₙ_lv, Eₘₐₓ_lv, τ, τₑₛ_lv, τₑₚ_lv, Eshift) # Left Ventricle pressure
    du[2] =
        (Qsv - Qmv) * Elastance_a(t, Eₘᵢₙ_la, Eₘₐₓ_la, τ, τₑₛ_la, τₑₚ_la) +
        (pLA / Elastance_a(t, Eₘᵢₙ_la, Eₘₐₓ_la, τ, τₑₛ_la, τₑₚ_la)) *
        DShiElastance_a(t, Eₘᵢₙ_la, Eₘₐₓ_la, τ, τₑₛ_la, τₑₚ_la) # Left Atrium pressure
    du[3] = (Qav - Qs) / Csa # Systemic Arterial pressure
    du[4] = (Qs - Qsv) / Csv # Systemic Venous pressure
    du[5] = Qmv - Qav # LV volume
    du[6] = Qsv - Qmv # LA volume
    du[7] = Valve(Zao, du[1] - du[3], pLV- psa)  # AV flow
    du[8] = Valve(Rmv, du[2] - du[1], pLA - pLV)  # MV flow
    du[9] = (du[3] - du[4]) / Rs # Systemic flow
    du[10] = (du[4] - du[2]) / Rsv # Venous flow
end

# Making ODE matrix -> one_chamber_sol
ode_problem = ODEProblem(NIK_2ch!, u0, tspan, ode_params)
ode_sol = solve(ode_problem, Vern7(); saveat = tsteps, reltol = 1e-6, abstol = 1e-6)
two_chamber_sol = Matrix(Array(ode_sol)')

"""
t_data_full = range(0, 40, length = size(original_data, 1))
mask_model = new_tseps .>= 35.0
time_to_plot = new_tseps[mask_model]
ode_to_plot  = two_chamber_sol[mask_model, :]
data_to_plot  = original_data[151:end , :]

labels = [
    "pLV",
    "pLA",
    "psa",
    "psv",
    "Vlv",
    "Vla",
    "Qav",
    "Qmv",
    "Qs",
    "Qsv",
]
plots = [
    begin
        p = plot(
            time_to_plot,
            two_ode_to_plot[:, i],
            label = "2 CHAMBER",
            xlabel = "time",
            ylabel = labels[i],
            lw = 2
        )
        p = plot!(
            time_to_plot,
            one_ode_to_plot[:, i-1],
            label = "1 CHAMBER",
            xlabel = "time",
            ylabel = labels[i],
            lw = 2
        )
        p = plot!(
            time_to_plot,
            data_to_plot[:, i],
            label = "4 CHAMBER",
            xlabel = "time",
            ylabel = labels[i],
            lw = 2
        )
        p
    end
    for i in 5:5
]
plot(
    plots...,
    layout = (1, 1),
    size = (900, 800)
)

"""
# -------ORIGINAL DATA INSPECTION ------
training_steps = tsteps
function zero_mean_loss(pred_norm, nn_vars)
    l = 0
    for j in nn_vars
        der_pred = diff(pred_norm[:, j])
        net_drift = mean(der_pred)
        l += abs2(net_drift)
    end
    return l # higher loss weight
end
function data_loss(pred_norm, data_norm, data_vars) 
    l = 0.0
    for (k, j) in enumerate(data_vars)
        l += mean(abs2, pred_norm[:, j] .- data_norm[:, k])
    end
    return l
end
function data_1deriv_loss(pred_norm, data_norm, data_vars)
    dt_pred = training_steps[2] - training_steps[1]
    dt_data = dt_pred   # MUST match, since you aligned data earlier
    l = zero(eltype(pred_norm))
    for j in data_vars
        # First derivative (central diff)
        d1_pred = (pred_norm[3:end, j] .- pred_norm[1:end-2, j]) ./ (2 * dt_pred)
        d1_data = (data_norm[3:end, j] .- data_norm[1:end-2, j]) ./ (2 * dt_data)
        l += mean(abs2, d1_pred .- d1_data)
    end
    return l
end

function physics_loss(pred_mat, ode_f, physics_vars, time_steps)
    l = 0.0
    for (i, t) in enumerate(time_steps)
        u = @view pred_mat[i, :]
        du = similar(u)
        ode_f(du, u, nothing, t)   # du = f(u,t)
        l += mean(abs2.(du[physics_vars]))
    end
    return l / length(time_steps)
end

function data_2deriv_loss(pred_norm, data_norm, data_vars)
    dt_pred = training_steps[2] - training_steps[1]
    dt_data = dt_pred   # MUST match, since you aligned data earlier
    l = zero(eltype(pred_norm))
    for j in data_vars
        # Second derivative (curvature)
        d2_pred = (pred_norm[3:end, j] .- 2pred_norm[2:end-1, j] .+ pred_norm[1:end-2, j]) ./ dt_pred^2
        d2_data = (data_norm[3:end, j] .- 2data_norm[2:end-1, j] .+ data_norm[1:end-2, j]) ./ dt_data^2
        l += 0.2 * mean(abs2, d2_pred .- d2_data)
    end
    return l
end

zm_data = zero_mean_loss(original_data, nn_vars)
zm_ode = zero_mean_loss(two_chamber_sol, nn_vars)
data_loss_data = data_loss(original_data, original_data, nn_vars)
data_loss_ode = data_loss(two_chamber_sol, original_data, nn_vars)
physics_loss_ode = physics_loss(two_chamber_sol, ode_problem.f, 1:10, tsteps)
physics_loss_data = physics_loss(original_data, ode_problem.f, 1:10, tsteps)
firstderiv_loss_data = data_1deriv_loss(original_data, original_data, 1:10)
firstderiv_loss_ode = data_1deriv_loss(two_chamber_sol, original_data, 1:6)
secondderiv_loss_data = data_2deriv_loss(original_data, original_data, 1:10)
secondderiv_loss_ode = data_2deriv_loss(two_chamber_sol, original_data, 1:10)

# ----------------PINN----------------------
nn_vars = [1, 2, 3, 4, 5, 6]
tspan = (0.0, 7.0)
num_of_samples = 150
tsteps = range(6.0, 7.0, length = num_of_samples)
loaded_data = readdlm("/Applications/Desktop/CODE/Data_acquisition/data/original_data_new.txt")
extrap_original_data = Array{Float64}(loaded_data)[1:3000, :]
original_data = extrap_original_data[901:1050, :]

τₑₛ_lv, τₑₚ_lv, τₑₛ_la, τₑₚ_la = 0.3, 0.45, 0.25, 0.15
Rmv, Zao, Rs, Rsv, Csa, Csv, Eₘₐₓ_lv, Eₘᵢₙ_lv, Eₘₐₓ_la, Eₘᵢₙ_la = [0.017922484159469603, 0.0020117225646972656, 1.4935249987244605, 0.06007680416107178, 0.8106463789939881, 12.51143291592598, 4.030184215307236, 0.044497499465942385, 0.12638190388679504, 0.22449553459882735]
ode_params = [τₑₛ_lv, τₑₚ_lv, Rmv, Zao, Rs, Rsv, Csa, Csv, Eₘₐₓ_lv, Eₘᵢₙ_lv, τₑₛ_la, τₑₚ_la, Eₘₐₓ_la, Eₘᵢₙ_la]
u0 = [7.0, 7.0, 7.0, 7.0, 200.0, 60.0, 0.0, 0.0, 0.0, 0.0] 
Eshift = 0.0
τ = 1.0

U_MEAN = vec(mean(original_data, dims = 1))
U_STD = vec(std(original_data, dims = 1)) .+ 1e-6
data_norm = (original_data .- U_MEAN') ./ U_STD'

ode_problem = ODEProblem(NIK_2ch!, u0, tspan, ode_params)
ode_sol = solve(ode_problem, Vern7(); saveat = tsteps, reltol = 1e-6, abstol = 1e-6)
two_chamber_sol = Matrix(Array(ode_sol)')
pred_norm = (two_chamber_sol .- U_MEAN') ./ U_STD'

NN = Lux.Chain(
    Lux.Dense(10, 10, tanh),
    Lux.Dense(10, 10, tanh),
    Lux.Dense(10, 10, tanh),
    Lux.Dense(10, 10, tanh),
    Lux.Dense(10, 6),
)

function data_loss(pred, data, data_vars)
    return sum(mean(abs2, pred[:, j] .- data[:, j]) for j in data_vars)
end
data_loss_ode = data_loss(pred_norm, data_norm, nn_vars)

include("/Applications/Desktop/CODE/Thesis/PINNFuser.jl/src/lib_new.jl")
using .LibInfuserNew
trained_p, trained_st, losses, nn_history = LibInfuserNew.PINN_Infuser_new(
    ode_problem,
    NN,
    tsteps,
    original_data;
    early_stopping = true,
    nn_output_weight = 0.1,
    physics_weight = 1.0,
    learning_rate = 1e-3,
    reltol = 1e-6,
    abstol = 1e-6,
    dtmax = 1e-2,
    iters = 200,
    loss_logfile = "training_logs/loss_history.txt",
    nn_vars = [1, 2, 3, 4, 5, 6],   
    data_vars = [1, 2, 3, 4, 5, 6],
    physics_vars = [1, 5, 6, 7, 8, 9, 10],
)

# Making PINN matrix -> pinn_pred
tspan = (0.0, 7.0)
tsteps = range(0.0, 7.0, length = 7 * 150)
function pinn_ode!(du, u, trained_p, t)
    nn_output = NN(u, trained_p, trained_st)[1]
    ode_problem.f(du, u, nothing, t)
    for (k, i) in enumerate(nn_vars)
        du[i] += 0.1 * nn_output[k]
    end    
end

pinn_problem = ODEProblem((du, u, p, t) -> pinn_ode!(du, u, p, t), 
            ode_problem.u0,
            tspan,
            trained_p,
        ) 
solved_pinn = solve(
            pinn_problem, 
            Vern7(),
            saveat=tsteps,
            dtmax=1e-2,
            reltol=1e-6,
            abstol=1e-6
        )
pinn_pred = hcat(solved_pinn.u...)'

# LONGER TWO CHAMBER SOLUTION 
ode_problem = ODEProblem(NIK_2ch!, u0, tspan, ode_params)
ode_sol = solve(ode_problem, Vern7(); saveat = tsteps, reltol = 1e-6, abstol = 1e-6)
two_chamber_sol = Matrix(Array(ode_sol)')
extrap_original_data = extrap_original_data[751:750 + (7*150), :]

#--------PLOT NN DIFFERENCES-----------
using LinearAlgebra

function nn_derivative_contribution(f_base!, f_pinn!, u_sol_base, u_sol_pinn, tsteps)
    n_times, n_vars = size(u_sol_base)
    du_base = zeros(n_vars)
    du_pinn = zeros(n_vars)
    nn_contrib = zeros(n_vars)

    tmp = zeros(n_vars)

    for ti in 1:n_times
        # Base ODE derivative
        f_base!(tmp, u_sol_base[ti, :], tsteps[ti])
        du_base .= tmp

        # PINN derivative
        f_pinn!(tmp, u_sol_pinn[ti, :], tsteps[ti])
        du_pinn .= tmp


        # Add difference
        nn_contrib .+= du_pinn .- du_base
    end

    return nn_contrib
end

#--------PLOT-----------
new_tseps = range(0, 7, length = 7*150)

t_data_full = range(0, 7, length = size(extrap_original_data, 1))
mask_model = new_tseps .>= 2.0
time_to_plot = new_tseps[mask_model]
data_to_plot = extrap_original_data[mask_model, :]
ode_to_plot  = two_chamber_sol[mask_model, :]
pinn_to_plot = pinn_pred[mask_model, :]

labelss = [
    "pLV",
    "pLA",
    "psa",
    "psv",
    "Vlv",
    "Vla",
    "Qav",
    "Qmv",
    "Qs",
    "Qsv",
]
plots = [
    begin
        p = plot(
            time_to_plot[:],
            pinn_to_plot[:, i],
            label = "PINN",
            xlabel = "time",
            ylabel = labelss[i],
            lw = 2
        )
        plot!(
            p,
            time_to_plot[:],
            ode_to_plot[:, i],
            label = "ODE",
            lw = 2,
            ls = :dash
        )
        plot!(
            p,
            time_to_plot[:],
            data_to_plot[:, i],
            label = "Data",
            lw = 2,
            ls = :dash
        )
        p
    end
    for i in 1:10
]
plot(
    plots...,
    layout = (5, 2),
    size = (900, 800)
)

# PLOTING LOSS
n_epochs = length(losses)
epochs = 1:n_epochs

plot(epochs, losses,
     xlabel = "Epoch",
     ylabel = "Loss",
     title = "Training Loss over Epochs",
     lw = 2,
     marker = :circle,
     markersize = 3,
     legend = false)

# PLOTTING NN CONTRIBUTIONS
n_vars = length(nn_history[1,1])  # 6 in your case
n_epochs = size(nn_history, 1)
epochs = 1:n_epochs

using Plots
p = plot(layout = (2,3), size=(1200,800))  # 2 rows x 3 columns for 6 variables

for j in 1:n_vars
    # Extract history of variable j across epochs
    var_history = [nn_history[i,1][j] for i in 1:n_epochs]
    
    plot!(p[j], epochs, var_history,
          title = labelss[j],
          xlabel = "Epochs",
          ylabel = "NN Contribution",
          legend = true)
end

display(p)
    
# ------------DERIVATIVES--------------
# visualize du: du_mat
function compute_du(u, t)
    du = similar(u0)
    NIK!(du, u, params, t)
    return du
end
du_mat = zeros(3000, 7)
for (k, (u, t)) in enumerate(zip(ode_sol.u, ode_sol.t))
    du_mat[k, :] .= compute_du(u, t)
end

# visualize du + nn: du_nn_mat
function compute_du_nn(u, trained_p, st, t)
    du = similar(u0)
    NIK!(du, u, params, t)
    nn_out = NN(u, trained_p, st)[1]
    for (k,j) in enumerate(nn_vars)
        du[j] += nn_out[k] 
    end
    return du
end
du_nn_mat = zeros(1500, 7)
for (k, (u, t)) in enumerate(zip(solved_pinn.u, solved_pinn.t))
    du_nn_mat[k, :] .= compute_du_nn(u, trained_p, st, t)
end

difff = du_nn_mat .- du_mat
time_to_plot = new_tseps[1:3000]

du_mat_plot = du_mat[751:1500, :]
du_nn_mat_plot = du_nn_mat[751:1500, :]
diff_plot = difff[751:1500, :]

labels = [
    "pLV",
    "psa",
    "psv",
    "Vlv",
    "Qav",
    "Qmv",
    "Qs"
]
using Plots
plots = [
    begin

        p = plot(
            time_to_plot[:],
            du_mat[:, i],
            label = "DIFF",
            xlabel = "time",
            ylabel = labels[i],
            lw = 2
        )
        p
    end
    for i in 1:7
]
plot(
    plots...,
    layout = (4, 2),
    size = (900, 500)
)

sum_cycle_ode = sum(du_mat_plot[1:151, 1])
sum_cycle_pinn = sum(du_nn_mat_plot[1:151, 1])
sum_cycle_diff = sum(diff_plot[1:151, 1])



# -------------CHANGING U0-----------------
u0 = [6.0, 6.0, 6.0, 200.0, 0.0, 0.0, 0.0] # Initial conditions pLV, psa, psv, Vlv, Qav, Qmv, Qs
# Making PINN matrix -> pinn_pred
function pinn_ode!(du, u, p, t)
    NIK!(du, u, params, t) # original ODE problem params
    nn_output = NN(u, trained_p, st)[1] # NN correction
    for i in nn_vars
        du[i] += nn_output[i] 
    end
end
pinn_problem = ODEProblem(pinn_ode!, u0, tspan) 
solved_pinn = solve(
    pinn_problem,
    Vern7(),
    saveat = new_tseps,
    reltol = 1e-6,
    abstol = 1e-6,
    dtmax = 1e-3,
) # t, u vector of vectors

NNcorr = zeros(3000, 7)
for (k, u) in enumerate(ode_sol.u)
    NNcorr[k, 1:4] .= NN(u, trained_p, st)[1]
end

t_data_full = range(0, 20, length = 3000)

labels = [
    "pLV",
    "psa",
    "psv",
    "Vlv",
    "Qav",
    "Qmv",
    "Qs"
]
using Plots
plots = [
    begin
        p = plot(
            t_data_full,
            NNcorr[0:3000, i],
            label = "PINN",
            xlabel = "time",
            ylabel = labels[i],
            lw = 2
        )
        p
    end
    for i in 1:7
]
plot(
    plots...,
    layout = (4, 2),
    size = (900, 800)
)

"""

    function energy_balance_loss(pred_norm, data_norm)
        # Ensure the Stroke Work (Area of P-V loop) is preserved
        # Work ≈ Σ P * ΔV
        work_pred = sum(pred_norm[:, 1] .* diff([pred_norm[:, 4]; pred_norm[1, 4]]))
        work_data = sum(data_norm[:, 1] .* diff([data_norm[:, 4]; data_norm[1, 4]]))
        
        return 1e-3 * abs2(work_pred - work_data)
    end
    function negativity_loss(pred_norm)
        pred_sub = pred_norm[:, 4:6] # Vlv, Qav, Qmv
        l = sum(relu.(-pred_sub).^2)
        return 10 * l / length(pred_sub)
    end
    function data_1deriv_loss(pred_norm, data_norm, data_vars)
        dt_pred = training_steps[2] - training_steps[1]
        dt_data = dt_pred   # MUST match, since you aligned data earlier
        l = zero(eltype(pred_norm))
        for (k, j) in enumerate(data_vars)
            # First derivative (central diff)
            d1_pred = (pred_norm[3:end, j] .- pred_norm[1:end-2, j]) ./ (2 * dt_pred)
            d1_data = (data_norm[3:end, k] .- data_norm[1:end-2, k]) ./ (2 * dt_data)
            l += mean(abs2, d1_pred .- d1_data)
        end
        return 1e-2 * l
    end
    function periodic_loss(pred_norm) # careful if we are working with more than 1 cycle
        start_u = pred_norm[1, :]
        end_u   = pred_norm[end, :]
        return mean(abs2.(end_u - start_u))
    end

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

    function data_1deriv_loss(pred_norm, data_norm, data_vars)
        dt_pred = training_steps[2] - training_steps[1]
        dt_data = dt_pred   # MUST match, since you aligned data earlier
        l = zero(eltype(pred_norm))
        for j in data_vars
            # First derivative (central diff)
            d1_pred = (pred_norm[3:end, j] .- pred_norm[1:end-2, j]) ./ (2 * dt_pred)
            d1_data = (data_norm[3:end, j] .- data_norm[1:end-2, j]) ./ (2 * dt_data)
            l += mean(abs2, d1_pred .- d1_data)
        end
        return 1e-2 * l
    end

    function data_2deriv_loss(pred_norm, data_norm, data_vars)
        dt_pred = training_steps[2] - training_steps[1]
        dt_data = dt_pred   # MUST match, since you aligned data earlier
        l = zero(eltype(pred_norm))
        for j in data_vars
            # Second derivative (curvature)
            d2_pred = (pred_norm[3:end, j] .- 2pred_norm[2:end-1, j] .+ pred_norm[1:end-2, j]) ./ dt_pred^2
            d2_data = (data_norm[3:end, j] .- 2data_norm[2:end-1, j] .+ data_norm[1:end-2, j]) ./ dt_data^2
            l += 0.2 * mean(abs2, d2_pred .- d2_data)
        end
        return 1e-6 * l
    end

    function plot_solution(temp_sol; target_data = nothing, training_steps = nothing, variable_names = nothing,)
        t = temp_sol.t
        sol_mat = hcat(temp_sol.u...)'   # (time × variables)
        n_vars = size(sol_mat, 2)
        plt = plot(layout = (n_vars, 1), size = (900, 250 * n_vars), link = :x)
        for i in 1:n_vars
            plot!(plt[i], t, sol_mat[:, i], label = "PINN", linewidth = 2,)
            plot!(plt[i], t, target_data[:, i], label = "Data", linestyle = :dash, )
            ylabel!(plt[i], variable_names === nothing ? "Var $i" : variable_names[i])
            if i == 1
                title!(plt[i], "PINN Prediction vs Data")
            end
            if i == n_vars
                xlabel!(plt[i], "Time")
            end
        end
        display(plt)
    end
"""