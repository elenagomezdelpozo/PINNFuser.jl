include("../src/Parameters.jl")
using .ParametersMod: parameters

include("../src/Model.jl")
using .ModelMod: NIK_2ch

using OrdinaryDiffEq
using Plots
using DelimitedFiles
    
ode_prob_base = ODEProblem(ModelMod.NIK_2ch!, parameters.u0, parameters.tspan, parameters.ode_params)
ode_sol_base  = solve(ode_prob_base, Vern7();
    saveat = parameters.plot_time, reltol = 1e-6, abstol = 1e-6)
original_ode_pred = Matrix(Array(ode_sol_base)')   # (time × vars)

target_data = parameters.original_all_data_list[:]   # (time × vars × patients)

plots = [
    begin
        p = plot(;
            title  = parameters.labels[i],
            xlabel = "time",
            ylabel = parameters.units[i],
            #ylims  = parameters.ylims[i],
        )
        plot!(p, parameters.plot_time, original_ode_pred[:, i],
            label = "Original ODE", lw = 4)
        for j in 1:parameters.number_of_patients
            plot!(p, parameters.plot_time, target_data[j][:, i],
                label = "Targets", lw = 2, ls = :dot)
        end
        p
    end
    for i in 1:length(parameters.vars)
    ]
p1 = plot(
    plots...,
    layout = (3, 2),
    size = (900, 800)
)
display(p1)  
