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
i = parameters.i

ode_problem = ODEProblem(LibInfuser.NIK_2ch!, parameters.u0, parameters.tspan, parameters.ode_params)
ode_sol_base  = solve(ode_problem, Vern7(); saveat = parameters.tsteps,
                      dtmax = parameters.dtmax, reltol = 1e-6, abstol = 1e-6)
ode_mat_base  = Array(ode_sol_base)'

if i == 7 # all variables
    nn_vars = parameters.vars
    @info "Training on all variables with loss(es) $(parameters.active)"
elseif i == 1:6 # some variables
    nn_vars = [parameters.vars[i]]
    @info "Training variable(s) $(i) with loss(es) $(parameters.active)"
else
    error("Invalid variable index. Please choose an index between 1 and 7.")
end
n = parameters.n_neurons_per_layer
NN = Lux.Chain(
    Lux.Dense(n, n, tanh),
    Lux.Dense(n, n, tanh),
    Lux.Dense(n, n, tanh),
    Lux.Dense(n, n, tanh),
    Lux.Dense(n, length(nn_vars)),
)
trained_p, trained_st, losses, nn_history = LibInfuser.PINN_Infuser_f( 
    ode_problem,
    ode_mat_base,
    parameters.ode_params,
    NN,
    parameters.tsteps,
    parameters.original_data;
    nn_vars = nn_vars, 
    early_stopping = true,
    plotting = parameters.plotting
)
jldsave( "trainings/pinn_model_$(name)_$(i).jld2"; 
        trained_p = trained_p, 
        trained_st = trained_st, 
        losses = losses,
        nn_history = nn_history,
        )
@info "Model saved successfully to pinn_model_$(name)_$(i).jld2"