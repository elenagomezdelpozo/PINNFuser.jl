using OrdinaryDiffEq
using Lux, Plots, Zygote, Statistics, StableRNGs, ComponentArrays
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using Optim, Measures, BenchmarkTools
using DelimitedFiles, ForwardDiff
using Plots
include("/Applications/Desktop/CODE/Thesis/PINNFuser.jl/examples/ElenasTry/cvmodelelena.jl")
using .elenasimpleModel
import Main.elenasimpleModel: Elastance_v, Elastance_a, DShiElastance_v, DShiElastance_a, Valve




# Things that can be changed
tspan = (0.0, 40.0)
num_of_cycles = 6 # 1 cycle for training
τ = 1.0 # Cardiac cycle period
extrapolation_tspan = (0.0, 40.0)
# params = [0.3, 0.45, 0.012, 0.004, 1.01, 0.12, 1.3, 35.0, 2.5, 0.1, 0.9, 0.95, 0.25, 0.15]
params = [0.3, 0.45, 0.012, 0.004, 1.01, 0.12, 1.3, 25.0, 2.7, 0.08, 0.9, 0.95, 0.25, 0.15]
τₑₛ_lv, τₑₚ_lv, Rmv, Zao, Rs, Rsv, Csa, Csv, Eₘₐₓ_lv, Eₘᵢₙ_lv, τₑₛ_la, τₑₚ_la, Eₘₐₓ_la, Eₘᵢₙ_la = params
Eshift = 0.0
u0 = [12.0, # pLV
    10.0,   # pLA
    80.0,   # psa
    15.0,   # psv
    120.0,  # Vlv
    100.0,  # Vla
    0.0,    # Qav
    0.0,    # Qmv
    0.0,    # Qs
    0.0,    # Qsv
    ] 

# Do not change these
num_of_samples_per_cycle = 150
num_of_samples = num_of_samples_per_cycle * num_of_cycles 
tsteps = range(5.0, 5.0 + num_of_cycles * τ , length = num_of_samples)
# loaded_data = readdlm("/Applications/Desktop/CODE/Data_acquisition/data/original_data.txt") # NEW DATA ACQUISITION METHOD
loaded_data = readdlm("/Applications/Desktop/CODE/Data_acquisition/data/original_data_new.txt") # NEW DATA ACQUISITION METHOD
extrap_original_data = Array{Float64}(loaded_data)[1:Int(floor(extrapolation_tspan[2] * num_of_samples_per_cycle)), :]
original_data = extrap_original_data[751:750 + num_of_samples, :]
new_tseps = range(extrapolation_tspan[1], extrapolation_tspan[2], length = Int(floor(extrapolation_tspan[2] * num_of_samples_per_cycle)))

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

function NIK_new!(du, u, p, t)
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
ode_problem = ODEProblem(NIK_1ch!, u0, tspan, params)
ode_sol = solve(ode_problem, Vern7(); saveat = new_tseps, reltol = 1e-6, abstol = 1e-6)
two_chamber_sol = Matrix(Array(ode_sol)')

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
function zero_mean_loss_correct(f_mat, nn_vars)
    nt = 150
    cycles = 8
    integral = zeros(length(nn_vars), cycles)
    for i in nn_vars
        for j in 1:cycles
            integral[i, j] = f_mat[j*nt, i] - f_mat[(j-1)*nt + 1, i]
        end
    end
    return integral
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
print(size(original_data))
original_data_zm_loss = zero_mean_loss_correct(original_data, [1,2,3,4]) 
one_chamber_zm_loss = zero_mean_loss_correct(one_chamber, [1,2,3,4]) 
"""
# ----------------PINN----------------------
nn_vars = [1,2,3,4,5, 6]

NN = Lux.Chain(
    Lux.Dense(10, 10, tanh),
    Lux.Dense(10, 10, tanh),
    Lux.Dense(10, 10, tanh),
    Lux.Dense(10, 10, tanh),
    Lux.Dense(10, 6),
)
original_data_to_train = original_data[1:150,:]
training_steps = range(0, 1, length=size(original_data_to_train,1))
ode_problem = ODEProblem(NIK_new!, u0, tspan_to_train, params)

include("/Applications/Desktop/CODE/Thesis/PINNFuser.jl/src/lib_new.jl")
using .LibInfuserNew
trained_p, st = LibInfuserNew.PINN_Infuser_new(
    ode_problem,
    NN,
    params,
    training_steps,
    original_data_to_train;
    physics_weight = 1.0,
    learning_rate = 1e-3,
    reltol = 1e-6,
    abstol = 1e-6,
    dtmax = 1e-2,
    iters = 200,
    loss_logfile = "training_logs/loss_history.txt",
    nn_vars = [1, 2, 3, 4, 5, 6],
    data_vars = [1, 2, 3, 5, 6],
    physics_vars = [1, 5, 6, 7],
)

# Making PINN matrix -> pinn_pred
function pinn_ode!(du, u, p_NN, t)
    nn_output = nn(u, p_NN, st)[1]
    ode_problem.f(du, u, params, t)
    for (k, i) in enumerate(nn_vars)
        du[i] += nn_output_weight * nn_output[k]
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
pinn_pred = hcat(solved_pinn.u...)'

function periodic_loss(pred_pred) # careful if we are working with more than 1 cycle
    loss = zeros(7)
    loss[:] = pred_pred[1, :] .- pred_pred[end, :]
    return loss
end
periodic_loss_pinn = periodic_loss(pinn_pred)

#--------PLOT-----------
t_data_full = range(0, 20, length = size(original_data, 1))
mask_model = new_tseps .>= 1.0
time_to_plot = new_tseps[mask_model]
pinn_to_plot = pinn_pred[mask_model, :]
data_to_plot = original_data[mask_model, :]
ode_to_plot  = two_chamber_sol[mask_model, :]

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
            ode_to_plot[:, i],
            label = "PINN",
            xlabel = "time",
            ylabel = labels[i],
            lw = 2
        )
        """
        plot!(
            p,
            time_to_plot,
            ode_to_plot[:, i],
            label = "ODE",
            lw = 2,
            ls = :dash
        )
        plot!(
            p,
            time_to_plot,
            data_to_plot[:, i],
            label = "Data",
            lw = 2,
            ls = :dash
        )
        """
        p
    end
    for i in 1:10
]
plot(
    plots...,
    layout = (5, 2),
    size = (900, 800)
)


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
