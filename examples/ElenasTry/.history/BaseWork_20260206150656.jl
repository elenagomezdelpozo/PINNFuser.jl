using OrdinaryDiffEq
using Lux, Plots, Zygote, Statistics, StableRNGs, ComponentArrays
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using Optim, Measures, BenchmarkTools
using DelimitedFiles, ForwardDiff
import Main.elenasimpleModel: ShiElastance, DShiElastance, Valve


include("/Applications/Desktop/CODE/Thesis/PINNFuser.jl/examples/ElenasTry/cvmodelelena.jl")
using .elenasimpleModel

# Things that can be changed
tspan = (0.0, 10.0)
num_of_cycles = 1 # 1 cycle for training
τ = 1.0 # Cardiac cycle period
extrapolation_tspan = (0.0, 10.0)
params = [0.3, 0.45, 0.006, 0.033, 1.11, 1.13, 11.0, 1.5, 0.03]
τₑₛ, τₑₚ, Rmv, Zao, Rs, Csa, Csv, Eₘₐₓ, Eₘᵢₙ = params
u0 = [6.0, 6.0, 6.0, 200.0, 0.0, 0.0, 0.0] # Initial conditions pLV, psa, psv, Vlv, Qav, Qmv, Qs
Eshift = 0.0

# Do not change these
num_of_samples_per_cycle = 150
num_of_samples = num_of_samples_per_cycle * num_of_cycles 
tsteps = range(5.0, 5.0 + num_of_cycles * τ , length = num_of_samples)
loaded_data = readdlm("/Applications/Desktop/CODE/Thesis/PINNFuser.jl/examples/OneChamberModelCVS/original_data.txt")
extrap_original_data = Array{Float64}(loaded_data)[1:Int(floor(extrapolation_tspan[2] * num_of_samples_per_cycle)), :]
original_data = extrap_original_data[751:750 + num_of_samples, :]
new_tseps = range(extrapolation_tspan[1], extrapolation_tspan[2], length = Int(floor(extrapolation_tspan[2] * 150)))

NN = Lux.Chain(
    Lux.Dense(7, 10, tanh),
    Lux.Dense(10, 10, tanh),
    Lux.Dense(10, 10, tanh),
    Lux.Dense(10, 10, tanh),
    Lux.Dense(10, 4),
)

function NIK!(du, u, p, t)
    pLV, psa, psv, Vlv, Qav, Qmv, Qs = u

    du[1] =
        (Qmv - Qav) * ShiElastance(t, Eₘᵢₙ, Eₘₐₓ, τ, τₑₛ, τₑₚ, Eshift) +
        pLV / ShiElastance(t, Eₘᵢₙ, Eₘₐₓ, τ, τₑₛ, τₑₚ, Eshift) *
        DShiElastance(t, Eₘᵢₙ, Eₘₐₓ, τ, τₑₛ, τₑₚ, Eshift)
    # Left Ventricle
    du[2] = (Qav - Qs) / Csa #Systemic arteries
    du[3] = (Qs - Qmv) / Csv # Venous
    du[4] = Qmv - Qav # LV volume
    du[5] = Valve(Zao, (du[1] - du[2]), u[1] - u[2])  # AV 
    du[6] = Valve(Rmv, (du[3] - du[1]), u[3] - u[1])  # MV
    du[7] = (du[2] - du[3]) / Rs # Systemic flow
end

ode_problem = ODEProblem(NIK!, u0, tspan, params)
include("/Applications/Desktop/CODE/Thesis/PINNFuser.jl/src/lib_new.jl")
using .LibInfuserNew
trained_p, st= LibInfuserNew.PINN_Infuser_new(
    ode_problem,
    NN,
    params,
    tsteps,
    original_data;
    physics_weight = 1.0,
    learning_rate = 1e-3,
    reltol = 1e-6,
    abstol = 1e-6,
    dtmax = 1e-2,
    iters = 100,
    loss_logfile = "training_logs/loss_history.txt",
    nn_vars = [1, 2, 3, 4],
    data_vars = [1, 2, 3, 4],
    physics_vars = [1, 5, 6, 7],
)

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
print(size(original_data))
original_data_zm_loss = zero_mean_loss_correct(original_data, [1,2,3,4]) 
one_chamber = one_chamber_sol[2801:2950, :]
one_chamber_zm_loss = zero_mean_loss_correct(one_chamber, [1,2,3,4]) 

# -----------USING THE TRAINED NN-------------
num_of_cycles = 1 # 1 cycle for training

# Do not change these
num_of_samples_per_cycle = 150
num_of_samples = num_of_samples_per_cycle * num_of_cycles 
tsteps = range(5.0, 5.0 + num_of_cycles * τ , length = num_of_samples)
tspan = (0.0, 10.0)
u0 = [6.0, 6.0, 6.0, 200.0, 0.0, 0.0, 0.0] # Initial conditions pLV, psa, psv, Vlv, Qav, Qmv, Qs
new_tseps = range(tspan[1], tspan[2], length = Int(floor(tspan[2] * 150)))
loaded_data = readdlm("/Applications/Desktop/CODE/Thesis/PINNFuser.jl/examples/OneChamberModelCVS/original_data.txt")
extrap_original_data = Array{Float64}(loaded_data)[1:Int(floor(20 * num_of_samples_per_cycle)), :]
original_data = extrap_original_data[751:750 + num_of_samples, :]


nn_vars = [1,2,3,4]

function NIK!(du, u, p, t)
    pLV, psa, psv, Vlv, Qav, Qmv, Qs = u

    du[1] =
        (Qmv - Qav) * ShiElastance(t, Eₘᵢₙ, Eₘₐₓ, τ, τₑₛ, τₑₚ, Eshift) +
        pLV / ShiElastance(t, Eₘᵢₙ, Eₘₐₓ, τ, τₑₛ, τₑₚ, Eshift) *
        DShiElastance(t, Eₘᵢₙ, Eₘₐₓ, τ, τₑₛ, τₑₚ, Eshift) 
    # Left Ventricle
    du[2] = (Qav - Qs) / Csa #Systemic arteries
    du[3] = (Qs - Qmv) / Csv # Venous
    du[4] = Qmv - Qav # LV volume
    du[5] = Valve(Zao, (du[1] - du[2]), u[1] - u[2])  # AV 
    du[6] = Valve(Rmv, (du[3] - du[1]), u[3] - u[1])  # MV
    du[7] = (du[2] - du[3]) / Rs # Systemic flow
end

# Making ODE matrix -> one_chamber_sol
ode_problem = ODEProblem(NIK!, u0, tspan, params)
ode_sol = solve(ode_problem, Vern7(); saveat = new_tseps, reltol = 1e-6, abstol = 1e-6)
one_chamber_sol = Matrix(Array(ode_sol)')

# Making PINN matrix -> pinn_pred
function pinn_ode!(du, u, p, t)
    ode_problem.f(du, u, params, t) # original ODE problem params
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
pinn_pred = hcat(solved_pinn.u...)'

function periodic_loss(pred_norm) # careful if we are working with more than 1 cycle
    start_u = pred_norm[1, :]
    end_u   = pred_norm[end, :]
    return mean(abs2.(end_u - start_u))
end


#--------PLOT-----------
t_data_full = range(0, 20, length = size(original_data, 1))
mask_model = new_tseps .>= 2.0
time_to_plot = new_tseps[mask_model]
pinn_to_plot = pinn_pred[mask_model, :]
data_to_plot = original_data[mask_model, :]
ode_to_plot  = one_chamber_sol[mask_model, :]

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
            time_to_plot,
            pinn_to_plot[:, i],
            label = "PINN",
            xlabel = "time",
            ylabel = labels[i],
            lw = 2
        )
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
        p
    end
    for i in 1:7
]
plot(
    plots...,
    layout = (4, 2),
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
    for i in nn_vars
        du[i] += nn_out[i] 
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
