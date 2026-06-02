__precompile__(false)  # Add this here
module PlotsMod

include("Model.jl")
using .ModelMod: NIK_2ch

include("Parameters.jl")
using .ParametersMod: parameters

export Plot_ODE, Plot_all_patients, Plot_target

using DelimitedFiles
using JLD2          
using Lux
using StableRNGs
using OrdinaryDiffEq
using Plots
using ComponentArrays

function Plot_ODE(name::String, ode_prob_base::ODEProblem)
    ode_sol       = solve(ode_prob_base, Vern7();
        saveat = parameters.plot_time, reltol = 1e-6, abstol = 1e-6)
    ode_pred = Matrix(Array(ode_sol)')
    plots = Vector{Plots.Plot}(undef, length(parameters.vars))
    # ── Plot ──────────────────────────────────────────────────────────────────
    @info "Plotting results for $(name)..."
    for i in 1:length(parameters.vars)
        p = plot(
            ode_sol.t,
            ode_pred[:, i],
            label = "ODE",
            xlabel = "time",
            ylabel = parameters.units[i],
            ylims = parameters.ylims[i],
            title = parameters.labels[i],
            lw = 2
        )

        target = parameters.extrap_original_data[
            Int(end - parameters.range_to_plot * parameters.num_of_samples_per_cycle + 1):end,
            i
        ]

        plot!(
            p,
            ode_sol.t,
            target,
            label = "Target",
            lw = 2,
            ls = :dash
        )

        plots[i] = p
    end

    p1 = plot(
        plots...,
        layout = (3, 2),
        size = (900, 800)
    )
    savefig(p1, "data_figures/$(name).png")

end #function

function Plot_all_patients(patient_odes::Vector)
    n_vars     = length(parameters.vars)
    n_patients = length(patient_odes)

    var_plots = [
        plot(
            title = parameters.labels[i],
            legend = false
        )
        for i in 1:n_vars
    ]

    # Plot all patient ODEs
    for (i, ode_prob) in enumerate(patient_odes)
        ode_sol  = solve(
            ode_prob,
            Vern7();
            saveat = parameters.plot_time,
            reltol = 1e-6,
            abstol = 1e-6
        )

        ode_pred = Matrix(Array(ode_sol)')

        for j in 1:n_vars
            plot!(
                var_plots[j],
                ode_sol.t,
                ode_pred[:, j],
                xlabel = "time",
                ylabel = parameters.units[j],
                lw = 1.5,
            )
        end
    end

    # Overlay target once
    idx1 = Int(
        size(parameters.extrap_original_data, 1) -
        parameters.range_to_plot * parameters.num_of_samples_per_cycle + 1
    )

    for j in 1:n_vars
        target = parameters.extrap_original_data[idx1:end, j]

        plot!(
            var_plots[j],
            parameters.plot_time,
            target,
            color = :black,
            lw = 2,
            label = "Target"
        )
    end

    p_final = plot(
        var_plots...,
        layout = (3, 2),
        size = (900, 800)
    )

    savefig(p_final, "data_figures/all_patients.png")
    @info "Saved all patients to data_figures/all_patients.png"
end

function Plot_target()
    plts = Vector{Plots.Plot}(undef, 6)
    for i in 1:6
        p = plot(
            title     = parameters.labels[i],
            xlabel    = "time",
            ylabel    = parameters.units[i],
            ylims     = parameters.ylims[i],
            legend    = false
        )
        plot!(
            p,
            parameters.extrap_original_data[Int(end-parameters.range_to_plot*parameters.num_of_samples_per_cycle+1):end, i],
            lw = 1.5,
        ) 
        plts[i] = p
    end
    
    fig = plot(
        plts...,
        layout      = (3, 2),
        size        = (900, 800)
        )
        
    savefig(fig, "data_figures/target_data.png")
    @info "Saved target data to data_figures/target_data.png"
end # function
end # module