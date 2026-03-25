using OrdinaryDiffEq
using Lux, Plots, Zygote, Statistics, StableRNGs, ComponentArrays
using Optimization, OptimizationOptimisers
using Optim, Measures, BenchmarkTools
using DelimitedFiles
using Plots, LinearAlgebra, JLD2
using SciMLSensitivity 
using Revise # to replace old PINN_Infuser

include("../src/Lib.jl")
using .LibInfuser

include("../src/Parameters.jl")
using .Parameters_module: parameters

name = "data"
active = ["data"]  
ode_problem = ODEProblem(NIK_2ch!, parameters.u0, parameters.tspan, parameters.ode_params)

for i in parameters.vars
    @info "Training variable(s) $(i) with loss(es) $(name)"
    nn_vars = [i]
    NN = Lux.Chain(
        Lux.Dense(parameters.n_neurons_per_layer, parameters.n_neurons_per_layer, tanh),
        Lux.Dense(parameters.n_neurons_per_layer, parameters.n_neurons_per_layer, tanh),
        Lux.Dense(parameters.n_neurons_per_layer, parameters.n_neurons_per_layer, tanh),
        Lux.Dense(parameters.n_neurons_per_layer, parameters.n_neurons_per_layer, tanh),
        Lux.Dense(parameters.n_neurons_per_layer, length(nn_vars)),
    )
    trained_p, trained_st, losses, nn_history = PINN_Infuser_funct( 
        ode_problem,
        parameters.params,
        NN,
        parameters.tsteps,
        parameters.original_data;
        active, 
        parameters.config,
        nn_vars = nn_vars, # for all variables
        learning_rate = 1e-4, # this could be changed
        dtmax = 1e-2, # this also matters, could be changed as well
        early_stopping = true,
    )
    jldsave("trainings/pinn_model_$(name)_$(i).jld2"; 
            trained_p = trained_p, 
            trained_st = trained_st, 
            losses = losses,
            nn_history = nn_history,
            )
    @info "Model saved successfully to pinn_model_$(name)_$(i).jld2"
    Saving_plots_funct(name, i)
end