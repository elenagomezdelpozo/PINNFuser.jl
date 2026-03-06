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
tspan = (0.0, 5.0)
num_of_cycles = 5 # 1 cycle for training
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
original_data = extrap_original_data[751:750 + num_of_samples, :]

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

trained_p, st= LibInfuserNew.PINN_Infuser_new(
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
    iters = 200,
    loss_logfile = "training_logs/loss_history.txt",
    nn_vars = [1, 2, 3, 4],
    data_vars = [1, 2, 3, 4, 5, 6, 7],
    physics_vars = [1, 5, 6, 7],
)

nn_vars = [1,2,3,4]
u0 = [3.8706866369176542, 85.54997478611195, 4.280132452062282, 129.02707151900782, 0.00014352086907499213, 68.24089433308623, 73.21607417482015]
ode_problem = ODEProblem(NIK!, u0, tspan, params)
function pinn_ode!(du, u, p, t)
    ode_problem.f(du, u, ode_problem.p, t) # original ODE problem
    nn_output = NN(u, trained_p, st)[1] # NN correction
    for i in nn_vars
        du[i] += nn_output[i]
    end
end
pinn_problem = ODEProblem(pinn_ode!, ode_problem.u0, tspan)
solved_pinn = solve(
    pinn_problem,
    Vern7(),
    saveat = range(tspan[1], tspan[2], length = num_of_samples),
    reltol = 1e-6,
    abstol = 1e-6,
    dtmax = 1e-3,
)
pred_mat = hcat(solved_pinn.u...)'

original_problem = ODEProblem(NIK!, ode_problem.u0, tspan)
solved_ode = solve(
    original_problem,
    Vern7(),
    saveat = range(tspan[1], tspan[2], length = num_of_samples),
    reltol = 1e-6,
    abstol = 1e-6,
    dtmax = 1e-3,
)
original_mat = hcat(solved_ode.u...)'

# visualize du + nn
function compute_du(u, t)
    du = similar(u0)
    pinn_ode!(du, u, nothing, t)
    return du
end
du_nn_mat = zeros(length(tsteps), length(u0))
for (k, (u, t)) in enumerate(zip(solved_pinn.u, solved_pinn.t))
    du_nn_mat[k, :] .= compute_du(u, t)
end

# visualize du
function compute_du(u, t)
    du = similar(u0)
    NIK!(du, u, nothing, t)
    return du
end
du_mat = zeros(length(tsteps), length(u0))
for (k, (u, t)) in enumerate(zip(solved_pinn.u, solved_pinn.t))
    du_mat[k, :] .= compute_du(u, t)
end
print(du_mat[:,1])
print(du_nn_mat[:,1])
print(du_nn_mat .- du_mat)
print(sum(du_nn_mat[:, 1]))
sum(du_nn_mat, dims=1)  # sum over time → one value per variable
u_start = solved_pinn.u[1][1]
u_end = solved_pinn.u[end][1]
@show u_end-u_start

ts = solved_pinn.t                      # time vector (length = 150)
us = hcat(solved_pinn.u...)'            # 150 × 7 matrix
us_original = hcat(solved_ode.u...)'            # 150 × 7 matrix

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
            ts,
            us[:, i],
            label = "PINN",
            xlabel = "time",
            ylabel = labels[i],
            lw = 2
        )

        plot!(
            p,
            ts,
            us_original[:, i],
            label = "Original",
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

print(us_original[100, :] .- us_original[250, :])