using OrdinaryDiffEq
using Lux, Plots, Zygote, Statistics, StableRNGs, ComponentArrays
using Optimization, OptimizationOptimisers
using Optim, Measures, BenchmarkTools
using DelimitedFiles
using Plots, LinearAlgebra, JLD2
# using LuxCUDA          # GPU support for Lux
using SciMLSensitivity # provides InterpolatingAdjoint / ZygoteVJP

include("cvmodelelena.jl")
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

for i in 1:10
    trained_p, trained_st, losses, nn_history = LibInfuserNew.PINN_Infuser_new(
        ode_problem,
        ode_params,
        NN,
        tsteps,
        original_data;
        processor = "cpu",
        nn_output_weight = 1.0,
        data_weight = 1e-2,
        deriv_weight = 0.0,
        physics_weight = 0.0,   
        zm_weight = 1.0,
        learning_rate = 1e-4,
        iters = 200,
        nn_vars = [i],   
        data_vars = [1,2,3,4,5,6],
        physics_vars = [5,6,7,8,9,10],
        deriv_vars = [1,3,5,7,9],
        zm_vars = [i],
    )
    #Saving everything for later
    #trained_p_cpu = cpu(trained_p)
    #trained_st_cpu = Lux.cpu(trained_st)
    jldsave("trained_pinn_model_zm_$(i).jld2"; 
            trained_p = trained_p, 
            trained_st = trained_st, 
            losses = losses,
            nn_history = nn_history,
            )
    @info "Model saved successfully to trained_pinn_model_zm_$(i).jld2"
end
