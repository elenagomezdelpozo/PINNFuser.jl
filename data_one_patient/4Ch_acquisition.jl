using CellMLToolkit, ModelingToolkit, OrdinaryDiffEq
using DelimitedFiles, StableRNGs, Statistics
using Plots

include("../src/Parameters.jl")
using .ParametersMod: parameters

rng = StableRNG(5958)
ml = CellModel("data_many_patients/ModelMain.cellml")

prob = ODEProblem(ml, parameters.tspan)
main_sol = solve(prob, Tsit5(); saveat = parameters.tsteps, reltol = 1e-4, abstol = 1e-7, dtmax = 1e-2)

sys = ml.sys

# Data for 2 Chambers

data_to_save = hcat(
    main_sol[sys.LV.Pi], # pLV
    main_sol[sys.LA.Pi], # pLA
    main_sol[sys.Sas.Pi], # pSA
    main_sol[sys.Svn.Pi], # pSV has changed
    main_sol[sys.LV.V], # vLV
    main_sol[sys.LA.V], # vLA
    main_sol[sys.LV.Qo], # Qav
    main_sol[sys.LA.Qo], # Qmv
    main_sol[sys.Sat.Qo], # Qs
    main_sol[sys.Svn.Qo], # Qsv
)

writedlm("data_one_patient/original_data_2Ch.txt", data_to_save)
loaded_data = readdlm("data_one_patient/original_data_2Ch.txt") 
data_to_plot = Array{Float64}(loaded_data)[Int(floor(parameters.tspan[1] * parameters.num_of_samples_per_cycle))+1:Int(floor(parameters.tspan[2] * parameters.num_of_samples_per_cycle)), :] 
mask_model = parameters.tsteps .>= parameters.plot_time[1]
time_to_plot = parameters.time_to_plot
two_chamber_sol = readdlm("data_one_patient/two_chamber_ode.txt") 
two_ode_to_plot = two_chamber_sol[mask_model, :]


labels = [
    "pLV",
    "pLA",
    "psa",
    "psv",
    "Vlv",
    "Vla",
    "Qav",
    "Qmv",
    "Qs",
    "Qsv",
]
ylims = [
    (0, 130),
    (4, 8),
    (50, 150),
    (5, 30),
    (0, 150),
    (0, 70),
    (0, 1500),
    (0, 800),
    (40, 100),
    (25, 100)
]
plots = [
    begin
        p = plot(
            time_to_plot,
            data_to_plot[:, i],
            label = "4 CHAMBER",
            xlabel = "time",
            ylabel = labels[i],
            ylims = ylims[i],
            lw = 2
        )
        plot!(time_to_plot,
            two_ode_to_plot[:, i],
            label = "2 CHAMBER",
            xlabel = "time",
            ylabel = labels[i],
            ylims = ylims[i],
            lw = 2
        )
        p
    end
    for i in 1:10
]
p1 = plot(
    plots...,
    layout = (5, 2),
    size = (900, 800)
)
display(p1)

"""
#P-V LOOP ATRIUM
p2 = plot(
    data_to_plot[:,6],
    data_to_plot[:,2],
    label = "PvsV",
    lw = 2
)
display(p2)

#P-V LOOP ATRIUM
p3 = plot(
    two_ode_to_plot[:,6],
    two_ode_to_plot[:,2],
    label = "PvsV",
    lw = 2
)
display(p3)

# Data for 1 Chamber
data_to_save = hcat(
    main_sol[sys.LV.Pi],
    main_sol[sys.Sas.Pi],
    main_sol[sys.Svn.Po],
    main_sol[sys.LV.V],
    main_sol[sys.LV.Qo],
    main_sol[sys.LA.Qo],
    main_sol[sys.Svn.Qi],
)
writedlm("data_one_patient/original_data_1Ch.txt", data_to_save)
"""