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

name = parameters.name
active = parameters.active  
ode_problem = ODEProblem(LibInfuser.NIK_2ch!, parameters.u0, parameters.tspan, parameters.ode_params)

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
    trained_p, trained_st, losses, nn_history = LibInfuser.PINN_Infuser_f( 
        ode_problem,
        parameters.ode_params,
        NN,
        parameters.tsteps,
        parameters.original_data;
        active, 
        parameters.config,
        nn_vars = nn_vars, 
        nn_output_weight = parameters.nn_output_weight,
        inisialisation = parameters.inisialisation,
        learning_rate = parameters.lr, 
        dtmax = parameters.dtmax,
        iters = parameters.iterations,
        early_stopping = true,
    )
    jldsave("trainings/pinn_model_$(name)_$(i).jld2"; 
            trained_p = trained_p, 
            trained_st = trained_st, 
            losses = losses,
            nn_history = nn_history,
            )
    @info "Model saved successfully to pinn_model_$(name)_$(i).jld2"
    Saving_plots_f(name, i)
end