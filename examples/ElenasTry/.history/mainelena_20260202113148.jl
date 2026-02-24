using OrdinaryDiffEq
using Lux, Plots, Zygote, Statistics, StableRNGs, ComponentArrays
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using Optim, Measures, BenchmarkTools
using DelimitedFiles, ForwardDiff
import Main.elenasimpleModel: ShiElastance, DShiElastance, Valve


include("/Applications/Desktop/CODE/Thesis/PINNFuser.jl/examples/ElenasTry/cvmodelelena.jl")
using .elenasimpleModel

include("/Applications/Desktop/CODE/Thesis/PINNFuser.jl/src/lib_new.jl")
using .LibInfuserNew

# Things that can be changed
tspan = (0.0, 7.0)
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

trained_p, trained_st, U_MEAN, U_STD = LibInfuserNew.PINN_Infuser_new(
    ode_problem,
    NN,
    params,
    tsteps,
    original_data;
    nn_output_weight = 0.1,
    physics_weight = 1.0,
    learning_rate = 1e-3,
    reltol = 1e-6,
    abstol = 1e-6,
    dtmax = 1e-2,
    iters = 400,
    loss_logfile = "training_logs/loss_history.txt",
    nn_vars = [1, 2, 3, 4],
    data_vars = [1, 2, 3, 4, 5, 6, 7],
    physics_vars = [1, 5, 6, 7],
)

# Save extrapolation data
include("/Applications/Desktop/CODE/Thesis/PINNFuser.jl/src/lib_new.jl")
using .LibInfuserNew

LibInfuserNew.PINN_Extrapolator_new(
    ode_problem,
    NN,
    (trained_p, trained_st, U_MEAN, U_STD),
    extrapolation_tspan,
    1500,
    "cvs_lib_extrapolation.txt",
    nn_output_weight = 0.1,
    reltol = 1e-6,
    abstol = 1e-6,
    dtmax = 1e-3,
    nn_vars = [1,2,3,4],
)

mkpath("plots")

pinn_pred = readdlm("cvs_lib_extrapolation.txt", ',', Float64)
ode_problem_extrap = ODEProblem(NIK!, u0, extrapolation_tspan, params)

ode_sol =
    solve(ode_problem_extrap, Vern7(); saveat = new_tseps, reltol = 1e-6, abstol = 1e-6)
one_chamber_sol = Matrix(Array(ode_sol)')

labels = [
    "Four chamber pLV",
    "PINN pLV",
    "One chamber pLV",
    "Four chamber psa",
    "PINN psa",
    "One chamber psa",
    "Four chamber psv",
    "PINN psv",
    "One chamber psv",
    "Four chamber Vlv",
    "PINN Vlv",
    "One chamber Vlv",
    "Four chamber Qav",
    "PINN Qav",
    "One chamber Qav",
    "Four chamber Qmv",
    "PINN Qmv",
    "One chamber Qmv",
    "Four chamber Qs",
    "PINN Qs",
    "One chamber Qs",
]

mask_model = new_tseps .>= 5.0

t_data_full = range(extrapolation_tspan[1], extrapolation_tspan[2], length = size(extrap_original_data, 1))
mask_data = t_data_full .>= 5.0

time_to_plot = new_tseps[mask_model]
pinn_to_plot = pinn_pred[mask_model, :]
data_to_plot = extrap_original_data[mask_data, :]
ode_to_plot  = one_chamber_sol[mask_model, :]

LibInfuserNew.PINNPlotter_new.plot_PINN_results_new(
    pinn_to_plot,   # PINN
    data_to_plot,   # Data
    ode_to_plot,    # ODE
    labels,
    time_to_plot,
    time_to_plot,
    "Time [s]",
    "State value",
    "CVS: PINN vs One chamber vs Four_chamber",
    "/Applications/Desktop/CODE/Thesis/PINNFuser.jl/examples/ElenasTry/elenacvs_comparison_400iter.png",
)

LibInfuserNew.PINNPlotter_new.plot_loss(
    "training_logs/loss_history.txt";
    plotfile = "plots/elenaloss.png",
)        