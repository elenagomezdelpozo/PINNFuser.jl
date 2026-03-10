using OrdinaryDiffEq
using Lux, Plots, Zygote, Statistics, StableRNGs, ComponentArrays
using Optimization, OptimizationOptimisers
using Optim, Measures, BenchmarkTools
using DelimitedFiles
using Plots, LinearAlgebra, JLD2
# using LuxCUDA          # GPU support for Lux
using SciMLSensitivity # provides InterpolatingAdjoint / ZygoteVJP
gr()
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
# ── Load model ────────────────────────────────────────────────────────────
for i in 1:10
    data = load("Trainings/trained_pinn_model_zm_$(i).jld2")
    trained_st = data["trained_st"]
    trained_p = data["trained_p"]
    losses = data["losses"]
    nn_history = data["nn_history"]

    # Reconstruct clean NamedTuple parameter structure (avoids ReshapedArray bug)
    rng = StableRNG(5958)
    trained_p, _ = Lux.setup(rng, NN)
    trained_p = ComponentVector{Float64}(trained_p)
    trained_p .= Float64.(data["trained_p"])  # copy values into fresh contiguous memory
    @info "Model loaded."

    # ── Settings (must match training) ───────────────────────────────────────
    nn_output_weight = 1.0
    nn_vars = [1, 2, 3, 4, 5, 6]   # match what was used in training
    tspan   = (0.0, 7.0)
    tsteps  = range(0.0, 7.0, length = 7 * 150)

    # ── PINN ODE (exactly mirrors cpu_pinn_ode! from training) ────────────────
    function pinn_ode!(du, u, p, t)
        nn_output = NN(u, p, trained_st)[1]   # plain Vector, no reshape — matches training
        ode_problem.f(du, u, ode_params, t)
        for (k, i) in enumerate(nn_vars)
            du[i] += nn_output_weight * nn_output[k]
        end
    end

    # ── Solve ─────────────────────────────────────────────────────────────────
    pinn_problem = ODEProblem(
        (du, u, p, t) -> pinn_ode!(du, u, p, t),
        ode_problem.u0,
        tspan,
        trained_p,
    )
    solved_pinn = solve(pinn_problem, Vern7();
        saveat = tsteps, dtmax = 1e-2, reltol = 1e-6, abstol = 1e-6)
    pinn_pred = Matrix(Array(solved_pinn)')   # (time × vars)

    # ── Baseline ODE (no NN) ──────────────────────────────────────────────────
    ode_prob_base = ODEProblem(NIK_2ch!, ode_problem.u0, tspan, ode_params)
    ode_sol       = solve(ode_prob_base, Vern7();
        saveat = tsteps, reltol = 1e-6, abstol = 1e-6)
    ode_pred = Matrix(Array(ode_sol)')

    # ── Data ──────────────────────────────────────────────────────────────────
    new_tsteps   = range(0.0, 7.0, length = 7 * 150)
    original_data = extrap_original_data[751:(750 + 7*150), :]

    # ── Mask to plot only t >= 2 (skip transient) ────────────────────────────
    mask         = new_tsteps .>= 2.0
    time_to_plot = new_tsteps[mask]
    data_to_plot = original_data[mask, :]
    ode_to_plot  = ode_pred[mask, :]
    pinn_to_plot = pinn_pred[mask, :]

    # ── Plot ──────────────────────────────────────────────────────────────────

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
    p1 = plot(
        plots...,
        layout = (5, 2),
        title = "Equation $(i)",
        size = (900, 800)
    )
    savefig(p1, "Figures/DataZM_var$(i).png")

    # PLOTING LOSS
    n_epochs = length(losses)
    epochs = 1:n_epochs

    p2 = plot(epochs, losses,
        xlabel = "Epoch",
        ylabel = "Loss",
        title = "Training Loss over Epochs for Equation $(i)",
        lw = 2,
        marker = :circle,
        markersize = 3,
        legend = false)
    
    savefig(p2, "Figures/DataZM_var$(i)_loss.png")

    # PLOTTING NN HISTORY
    tsteps = range(0.0, 1.0, length = 150)
    p3 = plot(
        tsteps,
        nn_history[end],
        label = "NN History",
        xlabel = "time",
        ylabel = labelss[i],
        lw = 2,
        title = "NN History for Equation $(i)",
    )
    savefig(p3, "Figures/DataZM_var$(i)_nn.png")
end