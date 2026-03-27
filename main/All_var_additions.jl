using OrdinaryDiffEq
using Lux, Plots, Zygote, Statistics, StableRNGs, ComponentArrays
using Optimization, OptimizationOptimisers
using Optim, Measures, BenchmarkTools
using DelimitedFiles
using Plots, LinearAlgebra, JLD2
using SciMLSensitivity 

include("../src/Lib.jl")
using .LibInfuser

ENV["CONFIG_TYPE"] = "all"
include("../src/Parameters.jl")
using .ParametersMod: parameters

name = parameters.name
ode_problem = ODEProblem(LibInfuser.NIK_2ch!, parameters.u0, parameters.tspan, parameters.ode_params)

@info "Training all variables with loss(es) $(name)"
nn_vars = parameters.vars
NN = Lux.Chain(
    Lux.Dense(parameters.n_neurons_per_layer, parameters.n_neurons_per_layer, tanh),
    Lux.Dense(parameters.n_neurons_per_layer, parameters.n_neurons_per_layer, tanh),
    Lux.Dense(parameters.n_neurons_per_layer, parameters.n_neurons_per_layer, tanh),
    Lux.Dense(parameters.n_neurons_per_layer, parameters.n_neurons_per_layer, tanh),
    Lux.Dense(parameters.n_neurons_per_layer, length(nn_vars)),
)
trained_p, trained_st, losses, nn_history = LibInfuser.PINN_Infuser_f( 
    ode_problem,
    parameters.ode_params,
    NN,
    parameters.tsteps,
    parameters.original_data;
    nn_vars = nn_vars, 
    early_stopping = true,
    plotting = true
)
jldsave("/net/people/plgrid/plgelenagdelpozo/CV_0D_models/PINNFuser.jl/trainings/pinn_model_$(name)_all.jld2"; 
        trained_p = trained_p, 
        trained_st = trained_st, 
        losses = losses,
        nn_history = nn_history,
        )
@info "Model saved successfully to pinn_model_$(name)_all.jld2"