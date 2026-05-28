__precompile__(false)  # Add this here
module TestMod

include("Model.jl")
using .ModelMod: NIK_2ch

include("Parameters.jl")
using .ParametersMod: parameters

include("Generate_patients.jl")
using .PatientsMod: generate_patients

export Tester_f

using DelimitedFiles
using JLD2          # ← add this — provides load()
using Lux
using StableRNGs
using OrdinaryDiffEq
using Plots
using ComponentArrays
using Random


function Tester_f(name, ode_parameters; test = false)
    name = "pinn_$(parameters.number_of_patients)_$(name)_hpc"
    data = load(parameters.savepath)
    trained_st = data["trained_st"]
    trained_p = data["trained_p"]
    @info "Model loaded from $(parameters.savepath)"
    
    nn_vars = parameters.vars  

    # Reconstruct clean NamedTuple parameter structure (avoids ReshapedArray bug)
    rng = StableRNG(parameters.seed)
    trained_p, _ = Lux.setup(rng, parameters.NN)
    trained_p = ComponentVector{Float64}(trained_p)
    trained_p .= Float64.(data["trained_p"])  # copy values into fresh contiguous memory
    
    # ── Baseline ODE (no NN) ──────────────────────────────────────────────────
    all_original_odes = []
    for (i, ode_param) in enumerate(ode_parameters)
        ode_prob_base = ODEProblem(ModelMod.NIK_2ch!, parameters.u0, parameters.tspan, ode_param)
        ode_sol       = solve(ode_prob_base, Vern7();
            saveat = parameters.plot_time, reltol = 1e-6, abstol = 1e-6)
        ode_pred = Matrix(Array(ode_sol)')
        push!(all_original_odes, ode_pred)
    end
    
    # ── Solve PINNS ─────────────────────────────────────────────────────────────────
    all_pinns = []
    for (i, ode_param) in enumerate(ode_parameters)
        function pinn_ode!(du, u, trained_p, t)
            nn_output = parameters.NN(u, trained_p, trained_st)[1]   # plain Vector, no reshape — matches training
            ModelMod.NIK_2ch!(du, u, ode_param, t)
            for (k, i) in enumerate(nn_vars)
                du[i] += parameters.nn_output_weight * nn_output[k]
            end
        end
        pinn_problem = ODEProblem(
            (du, u, p, t) -> pinn_ode!(du, u, p, t),
            parameters.u0,
            parameters.tspan,
            trained_p,
        )
        solved_pinn = solve(pinn_problem, Vern7();
        saveat = parameters.plot_time, dtmax = parameters.dtmax, reltol = 1e-6, abstol = 1e-6)
        pinn_pred = Matrix(Array(solved_pinn)')   # (time × vars)
        push!(all_pinns, pinn_pred)
    end
    
    # ── Data ──────────────────────────────────────────────────────────────────
    original_data = parameters.extrap_original_data[1501:1500 + length(parameters.plot_time), :]

    # ── Plot ──────────────────────────────────────────────────────────────────
    @info "Plotting results for variable $(name)..."
    plots = [
        begin
            p = plot(;                          
                title  = parameters.labels[i],
                xlabel = "time",
                ylabel = parameters.units[i],
                #ylims  = parameters.ylims[i],
            )
            for (idx, pinn_pred) in enumerate(all_pinns)   
                plot!(p, parameters.plot_time, pinn_pred[:, i],
                    label = length(all_pinns) > 1 ? "PINN $idx" : "PINN",
                    lw = 2)
            end
            if test == true
                for (idx, original_ode) in enumerate(all_original_odes)   
                plot!(p, parameters.plot_time, original_ode[:, i],
                    label = length(all_original_odes) > 1 ? "Original ODE $idx" : "Original ODE",
                    lw = 2)
                end
            end            
            plot!(p, parameters.plot_time, original_data[:, i],
                label = "Data", lw = 2, ls = :dot)
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
        # PLOTING LOSS
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