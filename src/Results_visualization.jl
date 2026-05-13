module PlotsMod

include("Model.jl")
using .ModelMod: NIK_2ch

include("Parameters.jl")
using .ParametersMod: parameters

export Plots_model, Plot_ODE, Plot_all_patients

using DelimitedFiles
using JLD2          
using Lux
using StableRNGs
using OrdinaryDiffEq
using Plots
using ComponentArrays

function Plot_model(name)
    data = load("trainings/$(name).jld2")
    trained_st = data["trained_st"]
    trained_p = data["trained_p"]
    losses = data["losses"]
    @info "Model loaded from \"trainings/$(name).jld2\""
    
    nn_vars = parameters.vars  
    
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
        ModelMod.NIK_2ch!(du, u, parameters.ode_params, t)
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
        saveat = parameters.plot_time, dtmax = parameters.dtmax, reltol = 1e-6, abstol = 1e-6)
    pinn_pred = Matrix(Array(solved_pinn)')   # (time × vars)

    # ── Baseline ODE (no NN) ──────────────────────────────────────────────────
    ode_prob_base = ODEProblem(ModelMod.NIK_2ch!, parameters.u0, parameters.tspan, parameters.ode_params)
    ode_sol       = solve(ode_prob_base, Vern7();
        saveat = parameters.plot_time, reltol = 1e-6, abstol = 1e-6)
    ode_pred = Matrix(Array(ode_sol)')

    # ── Data ──────────────────────────────────────────────────────────────────
    original_data = parameters.extrap_original_data[1501:1500 + length(parameters.plot_time), :]

    # ── Plot ──────────────────────────────────────────────────────────────────
    @info "Plotting results for variable $(name)..."
    plots = [
        begin
            p = plot(
                parameters.plot_time[:],
                pinn_pred[:, i],
                label = "PINN",
                xlabel = "time",
                ylabel = parameters.labels[i],
                lw = 2
            )
            plot!(
                p,
                parameters.plot_time[:],
                ode_pred[:, i],
                label = "ODE",
                lw = 2,
                ls = :dash
            )
            plot!(
                p,
                parameters.plot_time[:],
                original_data[:, i],
                label = "Data",
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
        size = (900, 800)
    )
    savefig(p1, "figures/$(name).png")

    # PLOTING LOSS
    n_epochs = length(losses)
    epochs = 1:n_epochs

    p2 = plot(epochs, losses,
        xlabel = "Epoch",
        ylabel = "Loss",
        title = "Training Loss over Epochs for Equation",
        lw = 2,
        marker = :circle,
        markersize = 3,
        legend = false)
    
    savefig(p2, "figures/$(name)_loss.png")

end #function

function Plot_ODE(name::String, ode_prob_base::ODEProblem)
    ode_sol       = solve(ode_prob_base, Vern7();
        saveat = parameters.plot_time, reltol = 1e-6, abstol = 1e-6)
    ode_pred = Matrix(Array(ode_sol)')
    
    # ── Plot ──────────────────────────────────────────────────────────────────
    @info "Plotting results for $(name)..."
    plots = [
        plot(
            ode_sol.t[:],
            ode_pred[:, i],
            label = "ODE",
            lw = 2
        )
        for i in 1:length(parameters.vars)
    ]
    p1 = plot(
        plots...,
        layout = (3, 2),
        size = (900, 800)
    )
    savefig(p1, "data_all_patients/$(name).png")

end #function

function Plot_all_patients(patient_odes::Vector)
    n_vars    = length(parameters.vars)
    n_patients = length(patient_odes)
    colors    = palette(:tab10, n_patients)

    # One plot per variable, pre-initialized
    var_plots = [plot(title = string(parameters.vars[i]), legend = :topright) for i in 1:n_vars]

    for (i, ode_prob) in enumerate(patient_odes)
        ode_sol  = solve(ode_prob, Vern7();
            saveat = parameters.plot_time, reltol = 1e-6, abstol = 1e-6)
        ode_pred = Matrix(Array(ode_sol)')

        for j in 1:n_vars
            plot!(
                var_plots[j],
                ode_sol.t[:],
                ode_pred[:, j],
                label = "patient $i",
                lw    = 1.5,
                color = colors[i]
            )
        end
    end

    p_final = plot(
        var_plots...,
        layout = (3, 2),
        size   = (1200, 900)
    )
    savefig(p_final, "data_all_patients/all_patients.png")
    @info "Saved all patients to data_all_patients/all_patients.png"
end # function

end # module