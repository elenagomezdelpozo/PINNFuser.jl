using OrdinaryDiffEq
using Lux, Plots, Zygote, Statistics, StableRNGs, ComponentArrays
using Optimization, OptimizationOptimisers
using Optim, Measures, BenchmarkTools
using DelimitedFiles
using Plots, LinearAlgebra, JLD2
using SciMLSensitivity

include("../src/Lib.jl")

include("../src/Parameters.jl")
using .ParametersMod: parameters

include("../data/Creating_starting_ODE.jl")

patients_params, patients_odes = PatientsMod.generate_patients_odes(parameters.number_of_patients)

# INDIVIDUALLY PLOT EACH PATIENT BEFORE TRAINING
for (i, p_ode) in enumerate(patients_odes)
    LibInfuser.Plot_ODE("patient_$(i)", p_ode) 
end

# ALL PATIENTS TOGETHER BEFORE TRAINING
LibInfuser.Plot_all_patients(patients_odes)