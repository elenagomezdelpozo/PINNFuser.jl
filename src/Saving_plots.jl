module Savingplots_module

function Saving_plots_funct(name, i)
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

    labels = [
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
                ylabel = labels[i],
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
    tsteps = range(0.0, 1.0, length = 150)
    p3 = plot(
        tsteps,
        nn_history[end],
        label = "NN History",
        xlabel = "time",
        ylabel = labels[i],
        lw = 2,
        title = "NN History for Equation $(i)",
    )
    savefig(p3, "figures/$(name)_var$(i)_nn.png")

end #function
end # module