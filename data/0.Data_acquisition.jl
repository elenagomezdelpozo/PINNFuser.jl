using CellMLToolkit, ModelingToolkit, OrdinaryDiffEq
using DelimitedFiles, StableRNGs, Statistics

include("../src/Parameters.jl")
using .ParametersMod: parameters

include("../src/Lib.jl")
using .LibInfuser

rng = StableRNG(5958)

ml = CellModel("data/Model/ModelMain.cellml")

# Training range
tspan = (0.0, 40.0)
num_of_samples = 150
tsteps = range(tspan[1], tspan[2], length = Int(tspan[2] * num_of_samples))

prob = ODEProblem(ml, tspan)
main_sol = solve(prob, Tsit5(); saveat = tsteps, reltol = 1e-4, abstol = 1e-7, dtmax = 1e-2)

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

writedlm("data/target_data.txt", data_to_save)
loaded_data = readdlm("data/target_data.txt") # NEW DATA ACQUISITION METHOD
LibInfuser.Plot_target()