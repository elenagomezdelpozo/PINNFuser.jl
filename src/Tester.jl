module TestMod

include("Model.jl")
using .ModelMod: NIK_2ch!

include("Parameters.jl")
using .ParametersMod: parameters

export Tester_f, generate_patients

using DelimitedFiles
using JLD2          # ← add this — provides load()
using Lux
using StableRNGs
using OrdinaryDiffEq
using Plots
using ComponentArrays
using Random

function generate_patients(n_samples::Int; seed::Int)
    rng = Random.seed!(seed)
    patients = Matrix{Float64}(undef, n_samples, 10)
    loguniform(a, b) = exp(rand(rng) * (log(b) - log(a)) + log(a))

    for i in 1:n_samples
        Rmv  = loguniform(0.005, 0.05)
        Zao  = loguniform(0.001, 0.01)
        Rs   = loguniform(0.5, 2.5)
        Rsv  = loguniform(0.02, 0.15)
        Csa  = loguniform(0.5, 2.5)
        Csv  = loguniform(5.0, 20.0)
        Emin_lv = loguniform(0.03, 0.15)
        Emax_lv = Emin_lv * rand(rng, 10.0:1.0:40.0)
        if Emax_lv < 1.5; Emax_lv = 1.5; end  # hard floor for systolic function
        Emin_la = loguniform(0.03, 0.12)
        Emax_la = Emin_la * rand(rng, 3.0:0.5:10.0)
        if Emax_lv <= Emax_la; Emax_lv = 1.2 * Emax_la; end  # LV must dominate

        patients[i, :] = [Rmv, Zao, Rs, Rsv, Csa, Csv, Emax_lv, Emin_lv, Emax_la, Emin_la]
    end
    return patients
end

function Tester_f(name, new_ode_parameters)
    i = parameters.i
    data = load("trainings/$(name)_$(i).jld2")
    trained_st = data["trained_st"]
    trained_p = data["trained_p"]
    losses = data["losses"]
    nn_history = data["nn_history"]
    @info "Model loaded from \"trainings/$(name)_$(i).jld2\""

    if i == 7
        nn_vars = parameters.vars  
    elseif i in 1:6
        nn_vars = [parameters.vars[i]]
    else 
        error("Invalid variable index: $(i). Must be 1-6 or 7 for all.")
    end
    @info "nn_vars: $(nn_vars)"

    # Reconstruct clean NamedTuple parameter structure (avoids ReshapedArray bug)
    NN = Lux.Chain(
        Lux.Dense(parameters.n_neurons_per_layer, parameters.n_neurons_per_layer, tanh),
        Lux.Dense(parameters.n_neurons_per_layer, parameters.n_neurons_per_layer, tanh),
        Lux.Dense(parameters.n_neurons_per_layer, parameters.n_neurons_per_layer, tanh),
        Lux.Dense(parameters.n_neurons_per_layer, parameters.n_neurons_per_layer, tanh),
        Lux.Dense(parameters.n_neurons_per_layer, length(nn_vars)),
    )
    rng = StableRNG(5958)
    trained_p, _ = Lux.setup(rng, NN)
    trained_p = ComponentVector{Float64}(trained_p)
    trained_p .= Float64.(data["trained_p"])  # copy values into fresh contiguous memory

    # ── PINN ODE (exactly mirrors pinn_ode! from training) ────────────────
    function pinn_ode!(du, u, trained_p, t)
        nn_output = NN(u, trained_p, trained_st)[1]   # plain Vector, no reshape — matches training
        ModelMod.NIK_2ch!(du, u, new_ode_parameters, t)
        for (k, i) in enumerate(nn_vars)
            du[i] += parameters.nn_output_weight * nn_output[k]
        end
    end
    
    # ── Solve ─────────────────────────────────────────────────────────────────
    pinn_problem = ODEProblem(
        (du, u, p, t) -> pinn_ode!(du, u, trained_p, t),
        parameters.u0,
        parameters.tspan,
        trained_p,
    )
    solved_pinn = solve(pinn_problem, Vern7();
        saveat = parameters.tsteps, dtmax = parameters.dtmax, reltol = 1e-6, abstol = 1e-6)
    pinn_pred = Matrix(Array(solved_pinn)')   # (time × vars)

    # ── Baseline ODE (no NN) ──────────────────────────────────────────────────
    ode_prob_base = ODEProblem(ModelMod.NIK_2ch!, parameters.u0, parameters.tspan, new_ode_parameters)
    ode_sol       = solve(ode_prob_base, Vern7();
        saveat = parameters.tsteps, reltol = 1e-6, abstol = 1e-6)
    ode_pred = Matrix(Array(ode_sol)')

    # ── Data ──────────────────────────────────────────────────────────────────
    original_data = parameters.extrap_original_data[1501:1500 + parameters.num_of_samples, :]

    # ── Mask to plot only t >= 2 (skip transient) ────────────────────────────
    mask         = parameters.tsteps .>= (first(parameters.tsteps) + 3 * parameters.τ)
    time_to_plot = parameters.tsteps[mask]
    data_to_plot = original_data[mask, :]
    ode_to_plot  = ode_pred[mask, :]
    pinn_to_plot = pinn_pred[mask, :]

    # ── Plot ──────────────────────────────────────────────────────────────────
    @info "Plotting results for testing model \"$(name)\""
    plots = [
        begin
            p = plot(
                time_to_plot[:],
                pinn_to_plot[:, i],
                label = "PINN",
                xlabel = "time",
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
            p
        end
        for i in 1:length(parameters.vars)
    ]
    p1 = plot(
        plots...,
        layout = (3, 2),
        title = "Equation $(i)",
        size = (900, 800)
    )
    savefig(p1, "figures/testing_$(name).png")

end #function
end # module