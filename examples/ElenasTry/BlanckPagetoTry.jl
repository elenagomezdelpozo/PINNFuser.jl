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

tspan = (0.0, 7.0)
num_of_samples = 150 * 6 # 1 cardiac cycle for training
tsteps = range(5.0, 11.0, length = num_of_samples)

extrapolation_tspan = (0.0, 20.0)

loaded_data = readdlm("/Applications/Desktop/CODE/Thesis/PINNFuser.jl/examples/OneChamberModelCVS/original_data.txt")
extrap_original_data = Array{Float64}(loaded_data)[1:Int(floor(extrapolation_tspan[2] * 150)), :]
original_data = extrap_original_data[751:750 + num_of_samples, :]

u0 = [6.0, 6.0, 6.0, 200.0, 0.0, 0.0, 0.0]
params = [0.3, 0.45, 0.006, 0.033, 1.11, 1.13, 11.0, 1.5, 0.03]
τₑₛ, τₑₚ, Rmv, Zao, Rs, Csa, Csv, Eₘₐₓ, Eₘᵢₙ = params
Eshift = 0.0
τ = 1.0 # Cardiac cycle period

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

E_elastic = function (u)
            pLV, psa, psv, Vlv, Qav, Qmv, Qs = u
            elastic_energy = 0.5 * (ShiElastance(0.0, Eₘᵢₙ, Eₘₐₓ, τ, τₑₛ, τₑₚ, Eshift) * Vlv^2)
        return elastic_energy 
    end

period = 1.0  # Cardiac cycle period
dt = tsteps[2] - tsteps[1]  # assuming uniform time steps
t0 = first(tsteps)
t_end = last(tsteps)
n_periods = Int(floor((t_end - t0) / period))
println(n_periods)

l = 0.0
l_energy = 0.0

energy_function = function (u)
            pLV, psa, psv, Vlv, Qav, Qmv, Qs = u
            elastic_energy = 0.5 * (ShiElastance(0.0, Eₘᵢₙ, Eₘₐₓ, τ, τₑₛ, τₑₚ, Eshift) * Vlv^2)
        return elastic_energy 
    end

k=0
t_start = t0 + k * period
t_finish = t_start + period
println("Period ", k+1, ": from ", t_start, " to ", t_finish)
i0 = findfirst(t -> t ≥ t_start, tsteps)
i1 = findlast(t -> t ≥ t_finish, tsteps)
println("Indices: i0 = ", i0, ", i1 = ", i1)

target_segment = original_data[i0:i1, :]
target_energy = [energy_function(u) for u in eachrow(target_segment)]
ref_energy_area = sum(target_energy) * dt
println("Reference energy area: ", ref_energy_area)
# --- Normalized loss ---
norm_factor = abs(ref_energy_area) + 1e-6
println("Normalization factor: ", norm_factor)

num_of_equations = size(original_data, 2)

function plot_equation(idx; plt_title = nothing)
    plot(
        original_data[:, idx],
        seriestype = :line,
        titlefontsize = 10,
    )
end

all_plots = [plot_equation(i) for i = 1:num_of_equations]
cols = 3
rows = ceil(Int, num_of_equations / cols)
plot_width = cols * 600
plot_height = rows * 400

p = plot(
    all_plots...,
    layout = (rows, cols),
    size = (plot_width, plot_height),
)
