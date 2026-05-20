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

patients_params, patients_odes = LibInfuser.generate_patients(parameters.number_of_patients, seed = 42);

# INDIVIDUALLY PLOT EACH PATIENT BEFORE TRAINING
for (i, p_ode) in enumerate(patients_odes)
    LibInfuser.Plot_ODE("patient_$(i)", p_ode) 
end

# ALL PATIENTS TOGETHER BEFORE TRAINING
LibInfuser.Plot_all_patients(patients_odes)

# TARGET MODEL
LibInfuser.Plot_target()