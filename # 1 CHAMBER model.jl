# 1 CHAMBER model

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
u0 = [6.0, 6.0, 6.0, 200.0, 0.0, 0.0, 0.0]
#. pLV, psa, psv, Vlv, Qav, Qmv, Qs
params = [0.3, 0.45, 0.012, 0.004, 1.01, 1.6, 20.5, 2.5, 0.1]
τₑₛ, τₑₚ, Rmv, Zao, Rs, Csa, Csv, Eₘₐₓ, Eₘᵢₙ = params
Eshift = 0.0
τ = 1.0


# Do not change these
num_of_samples_per_cycle = 150
num_of_samples = num_of_samples_per_cycle * num_of_cycles 
tsteps = range(5.0, 5.0 + num_of_cycles * τ , length = num_of_samples)
# loaded_data = readdlm("/Applications/Desktop/CODE/Data_acquisition/data/original_data.txt") # NEW DATA ACQUISITION METHOD
loaded_data = readdlm("/Applications/Desktop/CODE/Data_acquisition/data/original_data_new.txt") # NEW DATA ACQUISITION METHOD
extrap_original_data = Array{Float64}(loaded_data)[1:Int(floor(extrapolation_tspan[2] * num_of_samples_per_cycle)), :]
original_data = extrap_original_data[751:750 + num_of_samples, :]
new_tseps = range(extrapolation_tspan[1], extrapolation_tspan[2], length = Int(floor(extrapolation_tspan[2] * num_of_samples_per_cycle)))

function NIK_1ch!(du, u, p, t)
    pLV, psa, psv, Vlv, Qav, Qmv, Qs = u
    du[1] =
        (Qmv - Qav) * Elastance_v(t, Eₘᵢₙ, Eₘₐₓ, τ, τₑₛ, τₑₚ, Eshift) +
        (pLV / Elastance_v(t, Eₘᵢₙ, Eₘₐₓ, τ, τₑₛ, τₑₚ, Eshift)) *
        DShiElastance_v(t, Eₘᵢₙ, Eₘₐₓ, τ, τₑₛ, τₑₚ, Eshift)
    # Left Ventricle
    du[2] = (Qav - Qs) / Csa #Systemic arteries
    du[3] = (Qs - Qmv) / Csv # Venous
    du[4] = Qmv - Qav # LV volume
    du[5] = Valve(Zao, (du[1] - du[2]), u[1] - u[2])  # AV 
    du[6] = Valve(Rmv, (du[3] - du[1]), u[3] - u[1])  # MV
    du[7] = (du[2] - du[3]) / Rs # Systemic flow
end

# Making ODE matrix -> one_chamber_sol
ode_problem = ODEProblem(NIK_1ch!, u0, tspan, params)
ode_sol = solve(ode_problem, Vern7(); saveat = new_tseps, reltol = 1e-6, abstol = 1e-6)
one_chamber_sol = Matrix(Array(ode_sol)')

t_data_full = range(0, 40, length = size(original_data, 1))
mask_model = new_tseps .>= 35.0
time_to_plot = new_tseps[mask_model]
one_ode_to_plot  = one_chamber_sol[mask_model, :]
data_to_plot  = original_data[151:end , :]

labels = [
    "pLV",
    "psa",
    "psv",
    "Vlv",
    "Qav",
    "Qmv",
    "Qs",
]
plots = [
    begin
        p = plot(
            time_to_plot,
            one_ode_to_plot[:, i],
            label = "1 CHAMBER",
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
