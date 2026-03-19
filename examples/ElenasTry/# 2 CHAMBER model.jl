# 2 CHAMBER model


using OrdinaryDiffEq
using Lux, Plots, Zygote, Statistics, StableRNGs, ComponentArrays
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using Optim, Measures, BenchmarkTools
using DelimitedFiles, ForwardDiff
using Plots
include("/Applications/Desktop/CODE/PINNFuser.jl/examples/ElenasTry/cvmodelelena.jl")
using .elenasimpleModel
import Main.elenasimpleModel: Elastance_v, Elastance_a, DShiElastance_v, DShiElastance_a, Valve


# Things that can be changed
tspan = (0.0, 40.0)
num_of_cycles = 6 # 1 cycle for training
τ = 1.0 # Cardiac cycle period
extrapolation_tspan = (0.0, 40.0)
τₑₛ_lv, τₑₚ_lv, τₑₛ_la, τₑₚ_la = 0.3, 0.45, 0.25, 0.15
Rmv, Zao, Rs, Rsv, Csa, Csv, Eₘₐₓ_lv, Eₘᵢₙ_lv, Eₘₐₓ_la, Eₘᵢₙ_la = p_pred
u0 = [7.0, 7.0, 7.0, 7.0, 200.0, 60.0, 0.0, 0.0, 0.0, 0.0] 
#.    pLV, pLA, psa, psv, Vlv, Vla, Qav, Qmv, Qs, Qsv

Eshift = 0.0
τ = 1.0
"""
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
"""

# Do not change these
num_of_samples_per_cycle = 150
num_of_samples = num_of_samples_per_cycle * num_of_cycles 
tsteps = range(5.0, 5.0 + num_of_cycles * τ , length = num_of_samples)
# loaded_data = readdlm("/Applications/Desktop/CODE/Data_acquisition/data/original_data.txt") # NEW DATA ACQUISITION METHOD
loaded_data = readdlm("/Applications/Desktop/CODE/PINNFuser.jl/Data_acquisition/original_data_2Ch.txt") # NEW DATA ACQUISITION METHOD
extrap_original_data = Array{Float64}(loaded_data)[1:Int(floor(extrapolation_tspan[2] * num_of_samples_per_cycle)), :]
original_data = extrap_original_data[751:750 + num_of_samples, :]
new_tseps = range(extrapolation_tspan[1], extrapolation_tspan[2], length = Int(floor(extrapolation_tspan[2] * num_of_samples_per_cycle)))

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
ode_problem = ODEProblem(NIK_2ch!, u0, tspan, params)
ode_sol = solve(ode_problem, Vern7(); saveat = new_tseps, reltol = 1e-6, abstol = 1e-6)
two_chamber_sol = Matrix(Array(ode_sol)')

t_data_full = range(0, 40, length = size(original_data, 1))
mask_model = new_tseps .>= 35.0
time_to_plot = new_tseps[mask_model]
two_ode_to_plot  = two_chamber_sol[mask_model, :]
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
        plot!(time_to_plot,
            data_to_plot[:, i],
            label = "TARGET 4 CHAMBER",
            xlabel = "time",
            ylabel = labels[i],
            lw = 2
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
