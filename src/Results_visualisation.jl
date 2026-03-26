module PlotsMod

include("Model.jl")
using .ModelMod: NIK_2ch!

include("Parameters.jl")
using .ParametersMod: parameters

export Plots_f

function Plots_f(name, i)
    data = load("trainings/pinn_model_$(name)_$(i).jld2")
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
    nn_vars = parameters.vars[i]   # match what was used in training

    # ── PINN ODE (exactly mirrors pinn_ode! from training) ────────────────
    function pinn_ode!(du, u, p, t)
        nn_output = NN(u, p, trained_st)[1]   # plain Vector, no reshape — matches training
        ModelMod.NIK_2ch!(du, u, parameters.ode_params, t)
        for (k, i) in enumerate(nn_vars)
            du[i] += parameters.nn_output_weight * nn_output[k]
        end
    end

    # ── Solve ─────────────────────────────────────────────────────────────────
    pinn_problem = ODEProblem(
        (du, u, p, t) -> pinn_ode!(du, u, p, t),
        ode_problem.u0,
        parameters.plot_time,
        trained_p,
    )
    solved_pinn = solve(pinn_problem, Vern7();
        saveat = parameters.tsteps, dtmax = parameters.dtmax, reltol = 1e-6, abstol = 1e-6)
    pinn_pred = Matrix(Array(solved_pinn)')   # (time × vars)

    # ── Baseline ODE (no NN) ──────────────────────────────────────────────────
    ode_prob_base = ODEProblem(LibInfuser.NIK_2ch!, ode_problem.u0, tspan, ode_params)
    ode_sol       = solve(ode_prob_base, Vern7();
        saveat = parameters.plot_time, reltol = 1e-6, abstol = 1e-6)
    ode_pred = Matrix(Array(ode_sol)')

    # ── Data ──────────────────────────────────────────────────────────────────
    new_tsteps   = range(parameters.plot_time[1], parameters.plot_time[2], length = (parameters.plot_time[2] - parameters.plot_time[1]) * parameters.num_of_samples_per_cycle)
    original_data = extrap_original_data[15001:(1500 + parameters.plot_time[2] * parameters.num_of_samples_per_cycle), :]

    # ── Mask to plot only t >= 2 (skip transient) ────────────────────────────
    mask         = new_tsteps .>= 2.0
    time_to_plot = new_tsteps[mask]
    data_to_plot = original_data[mask, :]
    ode_to_plot  = ode_pred[mask, :]
    pinn_to_plot = pinn_pred[mask, :]

    # ── Plot ──────────────────────────────────────────────────────────────────

    plots = [
        begin
            p = plot(
                time_to_plot[:],
                pinn_to_plot[:, i],
                label = "PINN",
                xlabel = "time",
                ylabel = parameters.labels[i],
                ylims = parameters.ylims[i],
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
        title = "Equation $(i)",
        size = (900, 800)
    )
    savefig(p1, "figures/$(name)_var$(i).png")

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
    
    savefig(p2, "figures/$(name)_var$(i)_loss.png")

    # PLOTTING NN HISTORY
    tsteps = range(0.0, parameters.τ , length = parameters.num_of_samples_per_cycle)
    p3 = plot(
        tsteps,
        nn_history[end],
        label = "NN History",
        xlabel = "time",
        ylabel = parameters.labels[i],
        lw = 2,
        title = "NN History for Equation $(i)",
    )
    savefig(p3, "figures/$(name)_var$(i)_nn.png")

end #function
end # module