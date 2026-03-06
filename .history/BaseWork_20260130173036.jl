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

tspan = (0.0 , 5.0)
num_of_samples = 150 * 5 # 1 cardiac cycle for training
tsteps = range(5.0 , 6.0 , length = num_of_samples )
extrapolation_tspan = (0.0 , 5.0)
loaded_data = readdlm("/Applications/Desktop/CODE/Thesis/PINNFuser.jl/examples/OneChamberModelCVS/original_data.txt")
extrap_original_data = Array{Float64}(loaded_data)[1: Int(floor(20 * 150)), :]
original_data = extrap_original_data[751:750 + num_of_samples, :]

u0 = [6.0, 6.0, 6.0, 200.0, 0.0, 0.0, 0.0]
params = [0.3, 0.45, 0.006, 0.033, 1.11, 1.13, 11.0, 1.5, 0.03]
τₑₛ, τₑₚ, Rmv, Zao, Rs, Csa, Csv, Eₘₐₓ, Eₘᵢₙ = params
Eshift = 0.0
τ = 1.0 # Cardiac cycle period


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
new_tseps = range(extrapolation_tspan[1], extrapolation_tspan[2], length = Int(floor(extrapolation_tspan[2] * 150)))

mkpath("plots")

pinn_pred = readdlm("cvs_lib_extrapolation.txt", ',', Float64)
ode_problem_extrap = ODEProblem(NIK!, u0, extrapolation_tspan, params)

ode_sol = solve(ode_problem_extrap, Vern7(); saveat = new_tseps, reltol = 1e-6, abstol = 1e-6)
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

time_to_plot = new_tseps
data_to_plot = extrap_original_data
ode_to_plot  = one_chamber_sol

LibInfuserNew.PINNPlotter_new.plot_PINN_results_new(
    ode_to_plot,   # PINN
    ode_to_plot,   # Data
    data_to_plot,    # ODE
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