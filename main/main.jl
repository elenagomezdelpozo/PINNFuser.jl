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

include("../data/Creating_starting_ODE.jl")
using .PatientsMod: generate_patients_odes

patients_params, patients_odes = generate_patients_odes(parameters.number_of_patients)
name = parameters.name

nn_vars = parameters.vars
NN = Lux.Chain(
    Lux.Dense(length(parameters.u0), parameters.n_neurons_per_layer, tanh),
    Lux.Dense(parameters.n_neurons_per_layer, parameters.n_neurons_per_layer, tanh),
    Lux.Dense(parameters.n_neurons_per_layer, parameters.n_neurons_per_layer, tanh),
    Lux.Dense(parameters.n_neurons_per_layer, parameters.n_neurons_per_layer, tanh),
    Lux.Dense(parameters.n_neurons_per_layer, length(nn_vars)),
)

trained_p, trained_st, losses = LibInfuser.PINN_Infuser_f(
    patients_odes,
    patients_params,
    NN,
    parameters.original_data;
    nn_vars = nn_vars,
    early_stopping = true,
    plotting = true
)

jldsave(savepath = parameters.savepath;
        trained_p = trained_p,
        trained_st = trained_st,
        losses = losses
        )

@info "Model saved successfully to pinn_$(name).jld2"