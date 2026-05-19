using OrdinaryDiffEq
using Lux, Plots, Zygote, Statistics, StableRNGs, ComponentArrays
using Optimization, OptimizationOptimisers
using Optim, Measures, BenchmarkTools
using DelimitedFiles
using Plots, LinearAlgebra, JLD2
using SciMLSensitivity

include("../src/Lib.jl")
using .LibInfuser

include("../src/Parameters.jl")
using .ParametersMod: parameters

patients_params, patients_odes = LibInfuser.generate_patients(parameters.number_of_patients, seed = 42)
name = parameters.name

nn_vars = parameters.vars
NN = parameters.NN

trained_p, trained_st, losses = LibInfuser.PINN_Infuser_f(
    patients_odes,
    patients_params,
    NN,
    parameters.original_data;
    nn_vars = nn_vars,
    early_stopping = true,
    plotting = true
)

jldsave(parameters.savepath;
        trained_p = trained_p,
        trained_st = trained_st,
        losses = losses
        )

@info "Model saved successfully to pinn_$(name).jld2"