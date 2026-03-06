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
extrapolation_tspan = (0.0, 7.0)
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
    iters = 50,
    loss_logfile = "training_logs/loss_history.txt",
    nn_vars = [1, 2, 3, 4],
    data_vars = [1, 2, 3, 4, 5, 6, 7],
    physics_vars = [1, 5, 6, 7],
)

nn_vars = [1,2,3,4]

# Making ODE matrix -> one_chamber_sol
ode_problem_extrap = ODEProblem(NIK!, u0, extrapolation_tspan, params)
ode_sol =
    solve(ode_problem_extrap, Vern7(); saveat = new_tseps, reltol = 1e-6, abstol = 1e-6)
one_chamber_sol = Matrix(Array(ode_sol)')

# Making PINN matrix -> pinn_pred
function pinn_ode!(du, u, p, t)
    ode_problem.f(du, u, ode_problem.p, t) # original ODE problem params
    nn_output = NN(u, trained_p, st)[1] # NN correction
    for i in nn_vars
        du[i] += nn_output[i] * 0.1
    end
end
pinn_problem = ODEProblem(pinn_ode!, u0, tspan) 
solved_pinn = solve(
    pinn_problem,
    Vern7(),
    saveat = range(tspan[1], tspan[2], length = num_of_samples),
    reltol = 1e-6,
    abstol = 1e-6,
    dtmax = 1e-3,
) # t, u vector of vectors
pinn_pred = hcat(solved_pinn.u...)'





#--------PLOT-----------
t_data_full = range(0, 7, length = size(extrap_original_data, 1))
mask_data = t_data_full .>= 5.0

time_to_plot = new_tseps[mask_model]
pinn_to_plot = pinn_pred[mask_model, :]
data_to_plot = original_data[mask_data, :]
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
    NIK!(du, u, nothing, t)
    return du
end
du_mat = zeros(1500, 7)
for (k, (u, t)) in enumerate(zip(ode_sol.u, ode_sol.t))
    du_mat[k, :] .= compute_du(u, t)
end

# visualize du + nn: du_nn_mat
function pinn_ode!(du, u, p, t)
    ode_problem.f(du, u, ode_problem.p, t) # original ODE problem params
    nn_output = NN(u, trained_p, st)[1] # NN correction
    for i in nn_vars
        du[i] += nn_output[i] * 0.1
    end
end
function compute_du(u, pt)
    du = similar(u0)
    pinn_ode!(du, u, nothing, t)
    return du
end
du_nn_mat = zeros(1500, 7)
for (k, (u, t)) in enumerate(zip(solved_pinn.u, solved_pinn.t))
    du_nn_mat[k, :] .= compute_du(u, t)
end

mask_nodel = 750:1500
du_mat_plot = du_mat[751:1500, :]
du_nn_mat_plot = du_nn_mat[751:1500, :]

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
            du_mat_plot[:, i],
            label = "ODE",
            xlabel = "time",
            ylabel = labels[i],
            lw = 2
        )

        plot!(
            p,
            time_to_plot,
            du_nn_mat_plot[:, i],
            label = "PINN",
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