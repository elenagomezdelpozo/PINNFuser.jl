using OrdinaryDiffEq
using Lux, Plots, Zygote, Statistics, StableRNGs, ComponentArrays
using Optimization, OptimizationOptimisers
using Optim, Measures, BenchmarkTools
using DelimitedFiles, ForwardDiff
using Plots, LinearAlgebra, JLD2
include("../ElenasTry/cvmodelelena.jl")
using .elenasimpleModel
import Main.elenasimpleModel: Elastance_v, Elastance_a, DShiElastance_v, DShiElastance_a, Valve

p_pred = [0.017953664660453797,
0.0020019450187683105,
1.5139441400766374,
0.06003505110740662,
0.812327128648758,
12.51065507531166,
4.0488074094057085,
0.04460069417953491,
0.12569252401590347,
0.2247257262468338,
]

τ = 1.0 # Cardiac cycle period
τₑₛ_lv, τₑₚ_lv, τₑₛ_la, τₑₚ_la = 0.3, 0.45, 0.25, 0.15
Rmv, Zao, Rs, Rsv, Csa, Csv, Eₘₐₓ_lv, Eₘᵢₙ_lv, Eₘₐₓ_la, Eₘᵢₙ_la = p_pred 
Eshift = 0.0

ode_params = [τₑₛ_lv, τₑₚ_lv, Rmv, Zao, Rs, Rsv, Csa, Csv, Eₘₐₓ_lv, Eₘᵢₙ_lv, τₑₛ_la, τₑₚ_la, Eₘₐₓ_la, Eₘᵢₙ_la]
u0 = [7.0, 7.0, 7.0, 7.0, 200.0, 60.0, 0.0, 0.0, 0.0, 0.0] 

tspan = (5.0, 6.0)
tsteps = range(5.0, 6.0 , length = 150)
loaded_data = readdlm("../../Data_acquisition/original_data_2Ch.txt") # NEW DATA ACQUISITION METHOD
extrap_original_data = Array{Float64}(loaded_data)[1:Int(floor(40 * 150)), :] # 40 seconds of data, 150 samples per second
original_data = extrap_original_data[751:750 + 150, :]
extrap_tseps = range(0, 40, length = 40 * 150)

function NIK_2ch!(du, u, ode_params, t)
    τₑₛ_lv, τₑₚ_lv, Rmv, Zao, Rs, Rsv, Csa, Csv, Eₘₐₓ_lv, Eₘᵢₙ_lv, τₑₛ_la, τₑₚ_la, Eₘₐₓ_la, Eₘᵢₙ_la = ode_params
    pLV = u[1]; pLA = u[2]; psa = u[3]; psv = u[4]
    Vlv = u[5]; Vla = u[6]; Qav = u[7]; Qmv = u[8]
    Qs  = u[9]; Qsv = u[10]   
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
ode_problem = ODEProblem(NIK_2ch!, u0, tspan, ode_params)
ode_sol = solve(ode_problem, Vern7(); saveat = tsteps, reltol = 1e-6, abstol = 1e-6)
two_chamber_sol = Matrix(Array(ode_sol)')

nn_vars = [1, 2, 3, 4, 5, 6]
tspan = (0.0, 7.0)
num_of_samples = 150
tsteps = range(6.0, 7.0, length = num_of_samples)
loaded_data = readdlm("../../Data_acquisition/original_data_2Ch.txt")
extrap_original_data = Array{Float64}(loaded_data)[1:3000, :]
original_data = extrap_original_data[901:1050, :]

τₑₛ_lv, τₑₚ_lv, τₑₛ_la, τₑₚ_la = 0.3, 0.45, 0.25, 0.15
Rmv, Zao, Rs, Rsv, Csa, Csv, Eₘₐₓ_lv, Eₘᵢₙ_lv, Eₘₐₓ_la, Eₘᵢₙ_la = [0.017922484159469603, 0.0020117225646972656, 1.4935249987244605, 0.06007680416107178, 0.8106463789939881, 12.51143291592598, 4.030184215307236, 0.044497499465942385, 0.12638190388679504, 0.22449553459882735]
ode_params = [τₑₛ_lv, τₑₚ_lv, Rmv, Zao, Rs, Rsv, Csa, Csv, Eₘₐₓ_lv, Eₘᵢₙ_lv, τₑₛ_la, τₑₚ_la, Eₘₐₓ_la, Eₘᵢₙ_la]
u0 = [7.0, 7.0, 7.0, 7.0, 200.0, 60.0, 0.0, 0.0, 0.0, 0.0] 
Eshift = 0.0
τ = 1.0

U_MEAN = vec(mean(original_data, dims = 1))
U_STD = vec(std(original_data, dims = 1)) .+ 1e-6
data_norm = (original_data .- U_MEAN') ./ U_STD'

ode_problem = ODEProblem(NIK_2ch!, u0, tspan, ode_params)
ode_sol = solve(ode_problem, Vern7(); saveat = tsteps, reltol = 1e-6, abstol = 1e-6)
two_chamber_sol = Matrix(Array(ode_sol)')
pred_norm = (two_chamber_sol .- U_MEAN') ./ U_STD'

NN = Lux.Chain(
    Lux.Dense(10, 10, tanh),
    Lux.Dense(10, 10, tanh),
    Lux.Dense(10, 10, tanh),
    Lux.Dense(10, 10, tanh),
    Lux.Dense(10, 6),
)

include("../../src/lib_new.jl")
using .LibInfuserNew
trained_p, trained_st, losses, nn_history = LibInfuserNew.PINN_Infuser_new(
    ode_problem,
    ode_params,
    NN,
    tsteps,
    original_data;
    nn_output_weight = 0.1,
    learning_rate = 1e-4,
    iters = 200,
    nn_vars = [1, 2, 3, 4, 5, 6],   
    data_vars = [1, 2, 3, 4, 5, 6],
    physics_vars = [5, 6, 7, 8, 9, 10],
)
#Saving everything for later
trained_p_cpu = cpu_device()(trained_p)
trained_st_cpu = cpu_device()(trained_st)
jldsave("trained_pinn_model.jld2"; 
        trained_p = trained_p_cpu, 
        trained_st = trained_st_cpu, 
        losses = losses,
        nn_history = nn_history,
        )
@info "Model saved successfully to trained_pinn_model.jld2"
data = load("trained_pinn_model.jld2")
trained_p = data["trained_p"]
trained_st = data["trained_st"]
losses = data["losses"]
nn_history = data["nn_history"]
@info "Model loaded. Ready for inference."

# Making PINN matrix -> pinn_pred
tspan = (0.0, 7.0)
tsteps = range(0.0, 7.0, length = 7 * 150)
function pinn_ode!(du, u, p, t)
    nn_output, _ = NN(u, p, trained_st)
    ode_problem.f(du, u, ode_params, t)
    for (k, i) in enumerate(nn_vars)
        du[i] *= (1.0 + 0.1 * tanh(nn_output[k]))
    end    
end

pinn_problem = ODEProblem((du, u, p, t) -> pinn_ode!(du, u, p, t), 
            ode_problem.u0,
            tspan,
            trained_p,
        ) 
solved_pinn = solve(
            pinn_problem, 
            Vern7(),
            saveat=tsteps,
            dtmax=1e-2,
            reltol=1e-6,
            abstol=1e-6
        )
pinn_pred = hcat(solved_pinn.u...)'

using LinearAlgebra

function nn_derivative_contribution(f_base!, f_pinn!, u_sol_base, u_sol_pinn, tsteps)
    n_times, n_vars = size(u_sol_base)
    du_base = zeros(n_vars)
    du_pinn = zeros(n_vars)
    nn_contrib = zeros(n_vars)

    tmp = zeros(n_vars)

    for ti in 1:n_times
        # Base ODE derivative
        f_base!(tmp, u_sol_base[ti, :], tsteps[ti])
        du_base .= tmp

        # PINN derivative
        f_pinn!(tmp, u_sol_pinn[ti, :], tsteps[ti])
        du_pinn .= tmp


        # Add difference
        nn_contrib .+= du_pinn .- du_base
    end

    return nn_contrib
end

new_tseps = range(0, 7, length = 7*150)
original_data = extrap_original_data[751:(750 + 7*150), :]
ode_problem = ODEProblem(NIK_2ch!, u0, tspan, ode_params)
ode_sol = solve(ode_problem, Vern7(); saveat = tsteps, reltol = 1e-6, abstol = 1e-6)
two_chamber_sol = Matrix(Array(ode_sol)')

t_data_full = range(0, 7, length = size(original_data, 1))
mask_model = new_tseps .>= 2.0
time_to_plot = new_tseps[mask_model]
data_to_plot = original_data[mask_model, :]
ode_to_plot  = two_chamber_sol[mask_model, :]
pinn_to_plot = pinn_pred[mask_model, :]

labelss = [
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
            time_to_plot[:],
            pinn_to_plot[:, i],
            label = "PINN",
            xlabel = "time",
            ylabel = labelss[i],
            lw = 2
        )
        plot!(
            p,
            time_to_plot[:],
            ode_to_plot[:, i],
            label = "ODE",
            lw = 2,
            ls = :dash
        )
        plot!(
            p,
            time_to_plot[:],
            data_to_plot[:, i],
            label = "Data",
            lw = 2,
            ls = :dash
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

# PLOTING LOSS
n_epochs = length(losses)
epochs = 1:n_epochs

plot(epochs, losses,
     xlabel = "Epoch",
     ylabel = "Loss",
     title = "Training Loss over Epochs",
     lw = 2,
     marker = :circle,
     markersize = 3,
     legend = false)

# PLOTTING NN CONTRIBUTIONS
n_vars = length(nn_history[1,1])  # 6 in your case
n_epochs = size(nn_history, 1)
epochs = 1:n_epochs

using Plots
p = plot(layout = (2,3), size=(1200,800))  # 2 rows x 3 columns for 6 variables

for j in 1:n_vars
    # Extract history of variable j across epochs
    var_history = [nn_history[i,1][j] for i in 1:n_epochs]
    
    plot!(p[j], epochs, var_history,
          title = labelss[j],
          xlabel = "Epochs",
          ylabel = "NN Contribution",
          legend = true)
end

display(p)