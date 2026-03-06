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

tspan = (0.0 , 1.0)
num_of_samples = 200 * 1 # 1 cardiac cycle for training
tsteps = range(5.0 , 6.0 , length = num_of_samples )
extrapolation_tspan = (0.0 , 20.0)
loaded_data = readdlm("/Applications/Desktop/CODE/Thesis/PINNFuser.jl/examples/OneChamberModelCVS/original_data.txt")
extrap_original_data = Array{Float64}(loaded_data)[1: Int(floor(extrapolation_tspan[2] * 200)), :]
original_data = extrap_original_data[751:750 + num_of_samples, :]

u0_from_data = original_data[1, :]
params = [0.3, 0.45, 0.006, 0.033, 1.11, 1.13, 11.0, 1.5, 0.03]
τₑₛ, τₑₚ, Rmv, Zao, Rs, Csa, Csv, Eₘₐₓ, Eₘᵢₙ = params
Eshift = 0.0
τ = 1.0 # Cardiac cycle period

NN = Lux.Chain(
    Lux.Dense(7, 10, tanh),
    Lux.Dense(10, 10, tanh),
    Lux.Dense(10, 10, tanh),
    Lux.Dense(10, 10, tanh),
    Lux.Dense(10, 7; use_bias = false),
)

function NIK!(du, u, p, t)
    τₑₛ, τₑₚ, Rmv, Zao, Rs, Csa, Csv, Eₘₐₓ, Eₘᵢₙ = p
    pLV, psa, psv, Vlv, Qav, Qmv, Qs = u
    du[1] =
        (Qmv - Qav) * ShiElastance(t, Eₘᵢₙ, Eₘₐₓ, τ, τₑₛ, τₑₚ, Eshift) +
        pLV / (ShiElastance(t, Eₘᵢₙ, Eₘₐₓ, τ, τₑₛ, τₑₚ, Eshift)*1e-6) *
        DShiElastance(t, Eₘᵢₙ, Eₘₐₓ, τ, τₑₛ, τₑₚ, Eshift)
    # Left Ventricle
    du[2] = (Qav - Qs) / Csa #Systemic arteries     
    du[3] = (Qs - Qmv) / Csv # Venous
    du[4] = Qmv - Qav # LV volume
    u[5] = Valve(Zao, (u[1] - u[2]), u[2] - u[2])  # AV 
    u[6] = Valve(Rmv, (u[3] - u[1]), u[3] - u[1])  # MV
    du[7] = (du[2] - du[3]) / Rs # Systemic flow
    return du
end

ode_problem = ODEProblem(NIK!, u0_from_data, tspan, params)
include("/Applications/Desktop/CODE/Thesis/PINNFuser.jl/src/lib_new.jl")
using .LibInfuserNew
trained_p, trained_st, U_MEAN, U_STD = LibInfuserNew.PINN_Infuser_new(
    ode_problem,
    NN,
    params,
    tsteps,
    original_data;
    nn_output_weight = 0.001,
    physics_weight = 1.0,
    learning_rate = 0.001,
    reltol = 1e-6,
    abstol = 1e-6,
    dtmax = 1e-2,
    iters = 200,
    loss_logfile = "training_logs/loss_history.txt",
    nn_vars = [1, 2, 3, 4, 5, 6, 7],
    data_vars = [1, 2, 3, 4],
    physics_vars = [1, 5, 6, 7],
)

# Save extrapolation data
new_tseps = range(extrapolation_tspan[1], extrapolation_tspan[2], length = Int(floor(extrapolation_tspan[2] * 150)))
include("/Applications/Desktop/CODE/Thesis/PINNFuser.jl/src/lib_new.jl")
using .LibInfuserNew

LibInfuserNew.PINN_Extrapolator_new(
    ode_problem,
    NN,
    (trained_p, trained_st, U_MEAN, U_STD),
    params,
    extrapolation_tspan,
    Int(floor(extrapolation_tspan[2] * 150)),
    "cvs_lib_extrapolation.txt",
    nn_output_weight = 10.0,
    reltol = 1e-6,
    abstol = 1e-6,
    dtmax = 1e-2,
    nn_vars = [1],
)

mkpath("plots")

pinn_pred = readdlm("cvs_lib_extrapolation.txt", ',', Float64)
ode_problem_extrap = ODEProblem(NIK!, u0_from_data, extrapolation_tspan, params)

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
    ode_to_plot,   # Data
    ode_to_plot,    # ODE
    labels,
    time_to_plot,
    time_to_plot,
    "Time [s]",
    "State value",
    "CVS: PINN vs One chamber vs Four_chamber",
    "/Applications/Desktop/CODE/Thesis/PINNFuser.jl/examples/ElenasTry/elenacvs_comparison_slopeloss1.png",
)

LibInfuserNew.PINNPlotter_new.plot_loss(
    "training_logs/loss_history.txt";
    plotfile = "plots/elenaloss.png",
)        