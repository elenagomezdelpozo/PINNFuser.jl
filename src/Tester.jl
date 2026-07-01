__precompile__(false)  # Add this here
module TestMod

include("Model.jl")
using .ModelMod: NIK_2ch

include("Parameters.jl")
using .ParametersMod: parameters

export Tester_f

using DelimitedFiles
using JLD2          
using Lux
using StableRNGs
using OrdinaryDiffEq
using Plots
using ComponentArrays
using Random
using DataInterpolations: LinearInterpolation
"""
    Tester_f(name, new_patient_data; test = true)

Tests a trained single-ODE PINN corrector against a NEW target patient
(one not seen during training).

`new_patient_data` must be a (T × n_states) matrix of the target patient's
observed trajectory, sampled at `parameters.training_time` (same convention
used for training data, e.g. via `process_patient_data`).

The PINN derivative is:
    du = f(u_sim, ode_params, t) + nn_output_weight * NN([u_obs(t); u_sim(t)])
where u_obs(t) is built by interpolating `new_patient_data` over
`parameters.training_time`, exactly mirroring training.
"""
function Tester_f(name, new_patient_data::AbstractMatrix{Float64}; test = true)
    name = "pinn_$(parameters.number_of_patients)_$(name)_hpc"
    data = load(parameters.savepath)
    trained_st = data["trained_st"]
    @info "Model loaded from $(parameters.savepath)"

    nn_vars = parameters.vars
    n_states = length(parameters.u0)

    # Reconstruct clean NamedTuple parameter structure (avoids ReshapedArray bug)
    rng = StableRNG(parameters.seed)
    trained_p, _ = Lux.setup(rng, parameters.NN)
    trained_p = ComponentVector{Float64}(trained_p)
    trained_p .= Float64.(data["trained_p"])  # copy values into fresh contiguous memory

     # ── Build u_obs(t) interpolants for the new target patient ─────────────────
    t_obs = parameters.plot_time
    @assert size(new_patient_data, 1) == length(t_obs) "new_patient_data must have $(length(t_obs)) rows matching time length"
    @assert size(new_patient_data, 2) == n_states "new_patient_data must have $(n_states) columns matching state dimension"
 
    t_obs = Float32.(parameters.training_time)
    
    # ── Baseline ODE (no NN correction) ─────────────────────────────────────────
    ode_prob_base = ODEProblem(ModelMod.NIK_2ch!, parameters.u0, parameters.tspan, parameters.ode_params)
    ode_sol_base  = solve(ode_prob_base, Vern7();
        saveat = parameters.plot_time, reltol = 1e-6, abstol = 1e-6)
    original_ode_pred = Matrix(Array(ode_sol_base)')   # (time × vars)

    # ── Solve PINN-corrected ODE against the new patient ────────────────────────
    function pinn_ode!(du, u, p, t)
        u_obs = Zygote.ignore() do
            idx = clamp(searchsortedfirst(t_obs, Float32(t)), 1, length(t_obs))
            Float32[new_patient_data[idx, s] for s in 1:n_states]
        end        
        nn_input = vcat(u_obs, Float64.(u))             # (2*n_states,) — matches training
        nn_output = parameters.NN(nn_input, p, trained_st)[1]
        ModelMod.NIK_2ch!(du, u, parameters.ode_params, t)
        for (k, i) in enumerate(nn_vars)
            du[i] += parameters.nn_output_weight * nn_output[k]
        end
    end

    pinn_problem = ODEProblem(pinn_ode!, parameters.u0, parameters.tspan, trained_p)
    solved_pinn = solve(pinn_problem, Vern7();
        saveat = parameters.plot_time, dtmax = parameters.dtmax, reltol = 1e-6, abstol = 1e-6)
    pinn_pred = Matrix(Array(solved_pinn)')   # (time × vars)

    # ── Target patient data, restricted to plotting window ──────────────────────
    target_data = new_patient_data[end - length(parameters.plot_time) + 1:end, :]

    # ── Plot ──────────────────────────────────────────────────────────────────
    @info "Plotting results for $(name)..."
    plots = [
        begin
            p = plot(;
                title  = parameters.labels[i],
                xlabel = "time",
                ylabel = parameters.units[i],
                #ylims  = parameters.ylims[i],
            )
            plot!(p, parameters.plot_time, pinn_pred[:, i],
                label = "PINN", lw = 2)
            if test == true
                plot!(p, parameters.plot_time, original_ode_pred[:, i],
                    label = "Original ODE", lw = 2)
            end
            plot!(p, parameters.plot_time, target_data[:, i],
                label = "Target Patient", lw = 2, ls = :dot)
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

    if test == false
        # PLOTTING LOSS
        losses = data["losses"]
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
    end
end #function
end # module