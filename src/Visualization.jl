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
    
    # ── Plot ──────────────────────────────────────────────────────────────────
    @info "Plotting results for $(name)..."
    plots = [
        plot(
            ode_sol.t[:],
            ode_pred[:, i],
            label = "ODE",
            xlabel = "time",
            ylabel = parameters.units[i],
            ylims = parameters.ylims[i],
            title = parameters.labels[i],
            lw = 2
        )
        for i in 1:length(parameters.vars)
    ]
    p1 = plot(
        plots...,
        layout = (3, 2),
        size = (900, 800)
    )
    savefig(p1, "data_figures/$(name).png")

end #function

function Plot_all_patients(patient_odes::Vector)
    n_vars    = length(parameters.vars)
    n_patients = length(patient_odes)
    colors    = palette(:tab10, n_patients)

    # One plot per variable, pre-initialized
    var_plots = [plot(title = "$(parameters.labels[i])", legend = :topright) for i in 1:n_vars]

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
                xlabel = "time",
                ylabel = parameters.units[j],
                ylims = parameters.ylims[j],
                lw    = 1.5,
                color = colors[i]
            )
        end
    end

    p_final = plot(
        var_plots...,
        layout = (3, 2),
        size = (900, 800)
    )
    savefig(p_final, "data_figures/all_patients.png")
    @info "Saved all patients to data_figures/all_patients.png"
end # function

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