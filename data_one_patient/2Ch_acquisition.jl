using OrdinaryDiffEq
using Lux, Plots, Zygote, Statistics, StableRNGs, ComponentArrays
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using Optim, Measures, BenchmarkTools
using DelimitedFiles, ForwardDiff
using Plots

include("../src/Model.jl")
using .ModelMod: NIK_2ch!
include("../src/Parameters.jl")
using .ParametersMod: parameters

# Things that can be changed
#          Rmv,    Zao,   Rs,    Rsv,   Csa,    Csv,   Eₘₐₓ_lv, Eₘᵢₙ_lv, Eₘₐₓ_la, Eₘᵢₙ_la]
p_pred = [0.01,  0.002, 1.002,  0.02,  1.553,  30.90,   8.5,    0.07,   0.30,    0.019]
#.    pLV,   pLA,    psa,    psv,    Vlv,    Vla,  Qav, Qmv, Qs,  Qsv
u0 = [8.0,   7.0,    32.0,   22.73,  135.0,  85.0, 0.0, 0.0, 0.0, 0.0] 

# Making ODE matrix -> two_chamber_sol
ode_problem = ODEProblem(NIK_2ch!, u0, parameters.tspan, p_pred)
ode_sol = solve(ode_problem, Vern7(); saveat = parameters.tsteps, reltol = 1e-6, abstol = 1e-6)
two_chamber_sol = Matrix(Array(ode_sol)')

mask_model = parameters.tsteps .>= parameters.plot_time[1]
time_to_plot = parameters.time_to_plot
two_ode_to_plot  = two_chamber_sol[mask_model, :]
data_to_plot  = parameters.original_data

plots = [
    begin
        p = plot(
            time_to_plot,
            two_ode_to_plot[:, i],
            label = "2 CHAMBER",
            xlabel = "time",
            ylabel = parameters.labels[i],
            ylims = parameters.ylims[i],
            lw = 2
        )
        plot!(time_to_plot,
            data_to_plot[:, i],
            label = "TARGET 4 CHAMBER",
            lw = 2
        )
        p
    end
    for i in 1:6
]
p1 = plot(
    plots...,
    layout = (3, 2),
    size = (900, 800)
)
display(p1)

# load data
data_to_save = hcat(two_chamber_sol)
writedlm("two_chamber_ode.txt", data_to_save)