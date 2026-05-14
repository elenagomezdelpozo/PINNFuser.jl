module PlotsMod

include("Model.jl")
using .ModelMod: NIK_2ch!

include("Parameters.jl")
using .ParametersMod: parameters

export Plots_f

using DelimitedFiles
using JLD2          # ← add this — provides load()
using Lux
using StableRNGs
using OrdinaryDiffEq
using Plots
using ComponentArrays

function Plots_f(name, i)
    data = load("trainings/$(name)_all.jld2")
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
        saveat = parameters.tsteps, dtmax = parameters.dtmax, reltol = 1e-6, abstol = 1e-6)
    pinn_pred = Matrix(Array(solved_pinn)')   # (time × vars)

    # ── Baseline ODE (no NN) ──────────────────────────────────────────────────
    ode_prob_base = ODEProblem(ModelMod.NIK_2ch!, parameters.u0, parameters.tspan, parameters.ode_params)
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
    @info "Plotting results for variable $(name)..."
    plots = [
        begin
            p = plot(
                time_to_plot[:],
                pinn_to_plot[:, i],
                label = "PINN",
                xlabel = "time",
                ylabel = parameters.units[i],
                ylims = parameters.ylims[i],
                title = parameters.labels[i],
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
        title = "Training Loss over Epochs for Equation $(i)",
        lw = 2,
        marker = :circle,
        markersize = 3,
        legend = false)
    
    savefig(p2, "figures/$(name)_loss.png")
    if i == 7
        @info "No NN plot"
    else
        # PLOTTING NN HISTORY
        tsteps = range(0.0, parameters.τ , length = size(nn_history[end], 1))
        p3 = plot(
            tsteps,
            nn_history[end],
            label = "NN History",
            xlabel = "time",
            ylabel = parameters.labels[i],
            lw = 2,
            title = "NN History for Equation $(i)",
        )
        savefig(p3, "figures/$(name)_nn.png")
    end

end #function
end # module