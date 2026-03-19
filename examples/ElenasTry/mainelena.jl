using OrdinaryDiffEq
using Lux, Plots, Zygote, Statistics, StableRNGs, ComponentArrays
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using Optim, Measures, BenchmarkTools
using DelimitedFiles, ForwardDiff
import Main.elenasimpleModel: ShiElastance, DShiElastance, Valve


include("/Applications/Desktop/CODE/Thesis/PINNFuser.jl/examples/ElenasTry/cvmodelelena.jl")
using .elenasimpleModel

include("/Applications/Desktop/CODE/Thesis/PINNFuser.jl/src/lib.jl")
using .LibInfuser

# Training range
tspan = (0.0, 7.0)
num_of_samples = 300
tsteps = range(5.0, 7.0, length = num_of_samples)

extrapolation_tspan = (0.0, 20.0)

loaded_data = readdlm("/Applications/Desktop/CODE/Thesis/PINNFuser.jl/examples/OneChamberModelCVS/original_data.txt")
extrap_original_data = Array{Float64}(loaded_data)[1:Int(floor(extrapolation_tspan[2] * 150)), :]
original_data = extrap_original_data[751:1050, :]

u0 = [6.0, 6.0, 6.0, 200.0, 0.0, 0.0, 0.0]
params = [0.3, 0.45, 0.006, 0.033, 1.11, 1.13, 11.0, 1.5, 0.03]
τₑₛ, τₑₚ, Rmv, Zao, Rs, Csa, Csv, Eₘₐₓ, Eₘᵢₙ = params
Eshift = 0.0
τ = 1.0

NN = Lux.Chain(
    Lux.Dense(7, 10, tanh),
    Lux.Dense(10, 10, tanh),
    Lux.Dense(10, 10, tanh),
    Lux.Dense(10, 10, tanh),
    Lux.Dense(10, 7),
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

ode_problem = ODEProblem(NIK!, u0, tspan)
E_k = function (u)
        pLV, psa, psv, Vlv, Qav, Qmv, Qs = u
        kinetic_energy = 0.5 * (Qav^2 + Qmv^2 + Qs^2)
        return kinetic_energy
    end
E_elastic = function (u)
        pLV, psa, psv, Vlv, Qav, Qmv, Qs = u
        elastic_energy = 0.5 * ShiElastance(0.0, Eₘᵢₙ, Eₘₐₓ, τ, τₑₛ, τₑₚ, Eshift) * Vlv^2
        return elastic_energy
    end

trained_p, trained_st = LibInfuser.PINN_Infuser_new(
    ode_problem,
    NN,
    tsteps,
    original_data;
    energy_k_function = E_k,
    energy_elastic_function = E_elastic,
    early_stopping = true,
    nn_output_weight = 0.1,
    physics_weight = 1.0,
    optimizer = ADAM,
    learning_rate = 0.001,
    reltol = 1e-6,
    abstol = 1e-6,
    dtmax = 1e-2,
    iters = 200,
    loss_logfile = "training_logs/loss_history.txt",
    data_vars = [1, 2, 3, 4],
    physics_vars = [5, 6, 7],
)

# Save extrapolation data
new_tseps = range(extrapolation_tspan[1], extrapolation_tspan[2], length = Int(floor(extrapolation_tspan[2] * 150)))

LibInfuser.PINN_Extrapolator(
    ode_problem,
    NN,
    (trained_p, trained_st),
    extrapolation_tspan,
    Int(floor(extrapolation_tspan[2] * 150)),
    "cvs_lib_extrapolation.txt";
    nn_output_weight = 0.05,
    reltol = 1e-4,
    abstol = 1e-7,
    dtmax = 1e-2,
)

mkpath("plots")

pinn_pred = readdlm("cvs_lib_extrapolation.txt", ',', Float64)
ode_problem_extrap = ODEProblem(NIK!, u0, extrapolation_tspan)

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
ode_to_plot  = one_chamber_sol[mask_model, :]
data_to_plot = extrap_original_data[mask_data, :]

@assert size(pinn_to_plot, 1) == size(ode_to_plot, 1)
@assert size(data_to_plot, 2) == size(pinn_to_plot, 2)

LibInfuser.PINNPlotter.plot_PINN_results(
    pinn_to_plot,   # PINN
    data_to_plot,   # Data
    ode_to_plot,    # ODE
    labels,
    time_to_plot,
    time_to_plot,
    "Time [s]",
    "State value",
    "CVS: PINN vs One chamber vs Four_chamber",
    "elenacvs_comparison_withkineticenergy.png",
)

LibInfuser.PINNPlotter.plot_loss(
    "training_logs/loss_history.txt";
    plotfile = "plots/elenaloss.png",
)        