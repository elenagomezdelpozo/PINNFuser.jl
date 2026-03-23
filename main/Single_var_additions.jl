using OrdinaryDiffEq
using Lux, Plots, Zygote, Statistics, StableRNGs, ComponentArrays
using Optimization, OptimizationOptimisers
using Optim, Measures, BenchmarkTools
using DelimitedFiles
using Plots, LinearAlgebra, JLD2
using SciMLSensitivity # provides InterpolatingAdjoint / ZygoteVJP

if !isdefined(Main, :elenasimpleModel)
    include("/Applications/Desktop/CODE/PINNFuser.jl/main/cvmodelelena.jl")
    using .elenasimpleModel
    import Main.elenasimpleModel: Elastance_v, Elastance_a, DShiElastance_v, DShiElastance_a, Valve
end

function NIK_2ch!(du, u, p, t)
    pLV, pLA, psa, psv, Vlv, Vla, Qav, Qmv, Qs, Qsv = u
    Rmv, Zao, Rs, Rsv, Csa, Csv, Eₘₐₓ_lv, Eₘᵢₙ_lv, Eₘₐₓ_la, Eₘᵢₙ_la = p

    E_lv  = Elastance_v(t,  Eₘᵢₙ_lv, Eₘₐₓ_lv, τ, τₑₛ_lv, τₑₚ_lv, Eshift)
    dE_lv = DShiElastance_v(t, Eₘᵢₙ_lv, Eₘₐₓ_lv, τ, τₑₛ_lv, τₑₚ_lv, Eshift)
    E_la  = Elastance_a(t,  Eₘᵢₙ_la, Eₘₐₓ_la, τ, τₑₛ_la, τₑₚ_la)
    dE_la = DShiElastance_a(t, Eₘᵢₙ_la, Eₘₐₓ_la, τ, τₑₛ_la, τₑₚ_la)

    du[1] = (Qmv - Qav) * E_lv + (pLV / E_lv) * dE_lv # Left Ventricle pressure
    du[2] = (Qsv - Qmv) * E_la + (pLA / E_la) * dE_la # Left Atrium pressure
    du[3] = (Qav - Qs) / Csa # Systemic Arterial pressure
    du[4] = (Qs - Qsv) / Csv # Systemic Venous pressure
    du[5] = Qmv - Qav # LV volume
    du[6] = Qsv - Qmv # LA volume
    du[7] = Valve(Zao, du[1] - du[3], pLV- psa)  # AV flow
    du[8] = Valve(Rmv, du[2] - du[1], pLA - pLV)  # MV flow
    du[9] = (du[3] - du[4]) / Rs # Systemic flow
    du[10] = (du[4] - du[2]) / Rsv # Venous flow
end

NN = Lux.Chain(
    Lux.Dense(10, 10, tanh),
    Lux.Dense(10, 10, tanh),
    Lux.Dense(10, 10, tanh),
    Lux.Dense(10, 10, tanh),
    Lux.Dense(10, 1),
)

tspan = (0.0, 40.0)
num_of_samples_per_cycle = 150
num_of_cycles = 1
num_of_samples = num_of_samples_per_cycle * num_of_cycles
tsteps = range(39.0, 40.0, length = num_of_samples)

# ODE data
Eshift = 0.0
τ = 1.0
τₑₛ_lv, τₑₚ_lv, τₑₛ_la, τₑₚ_la = 0.3, 0.45, 0.92, 0.09
u0 = [8.0,   8.0,    30.0,   21.5,    130.0,  75.0, 0.0, 0.0, 0.0, 0.0] 
params = [0.013,  0.0020, 1.292,  0.070,  1.023,  10.90,  5.2,     0.0709,  0.20,    0.06]
Rmv, Zao, Rs, Rsv, Csa, Csv, Eₘₐₓ_lv, Eₘᵢₙ_lv, Eₘₐₓ_la, Eₘᵢₙ_la = params
ode_problem = ODEProblem(NIK_2ch!, u0, tspan, params)

# Target data
loaded_data = readdlm("/Applications/Desktop/CODE/PINNFuser.jl/data_acquisition/original_data_2Ch.txt")
extrapolation_tspan = (0.0, 40.0)
extrap_original_data = Array{Float64}(loaded_data)[1:Int(floor(extrapolation_tspan[2] * num_of_samples_per_cycle)), :]
original_data = extrap_original_data[1501:1500 + num_of_samples, :]

include("/Applications/Desktop/CODE/PINNFuser.jl/src/PINN_Infuser_new.jl")
using .PINNInfuser_new # name of module
name = "data"
active = ["data"]
config = (
    data_vars   = [1, 2, 3, 4, 5, 6],
    data_weight = 1.0,
    physics_vars = [1,2,3],
    physics_weight = 0.1,
    mass_conservation_weight = 1.0,
    zm_vars = [1, 2, 3, 4, 5, 6],
    zm_weight = 1.0,
    neg_vars = [5, 6],
    neg_weight = 10.0,
    firstderiv_vars = [1, 2, 3, 4, 5, 6],
    deriv_weight = 1.0,
    periodic_vars = [1,2,3,4,5,6],
    periodic_weight = 1.0
)

# Training and saving the resulting data (in Trainings)
for i in 1:6
    trained_p, trained_st, losses, nn_history = PINN_Infuser_new( # name of function
        ode_problem,
        params,
        NN,
        tsteps,
        original_data;
        active, 
        config,
        nn_vars = [i],
        nn_output_weight = 1.0,
        learning_rate = 1e-5,
        iters = 200
    )

    jldsave("trainings/trained_pinn_model_$(name)_$(i).jld2"; 
            trained_p = trained_p, 
            trained_st = trained_st, 
            losses = losses,
            nn_history = nn_history,
            )
    @info "Model saved successfully to trained_pinn_model_$(name)_$(i).jld2"
end

# Saving plots of the trainings (in Figures)
for i in 1:6
    data = load("trainings/trained_pinn_model_$(name)_$(i).jld2")
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
    nn_vars = [i]   # match what was used in training
    tspan   = (0.0, 7.0)
    tsteps  = range(0.0, 7.0, length = 7 * 150)

    # ── PINN ODE (exactly mirrors pinn_ode! from training) ────────────────
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
        "Vla"
    ]
    ylims = [
        (0, 130),
        (4, 10),
        (40, 140),
        (22,24),
        (30, 150),
        (20, 70)
    ]
    plots = [
        begin
            p = plot(
                time_to_plot[:],
                pinn_to_plot[:, i],
                label = "PINN",
                xlabel = "time",
                ylabel = labelss[i],
                ylims = ylims[i],
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
        for i in 1:6
    ]
    p1 = plot(
        plots...,
        layout = (3, 2),
        title = "Equation $(i)",
        size = (900, 800)
    )
    savefig(p1, "figures/Data$(name)_var$(i)_2.png")

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
    
    savefig(p2, "figures/Data$(name)_var$(i)_loss_2.png")

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
    savefig(p3, "figures/Data$(name)_var$(i)_nn_2.png")
end
