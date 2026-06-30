using OptimizationOptimisers: ADAM
using JLD2
using StaticArrays, OrdinaryDiffEq


include("../src/Lib.jl")
using .LibInfuser

include("../src/Parameters.jl")
using .ParametersMod: parameters

name = parameters.name

ode_problem = ODEProblem(LibInfuser.NIK_2ch, SA[parameters.u0...], parameters.tspan, parameters.ode_params)

nn_vars = parameters.vars
trained_p, trained_st, losses = LibInfuser.PINN_Infuser_f(
    ode_problem,
    parameters.ode_params,
    parameters.NN,
    parameters.original_training_data_list;
    nn_vars = nn_vars,
    optimizer = ADAM,
    reltol = Float32(1e-6),
    abstol = Float32(1e-6),
    early_stopping = true,
    plotting = parameters.plotting
)

jldsave(parameters.savepath;
        trained_p = trained_p,
        trained_st = trained_st,
        losses = losses
        )

@info "Model saved successfully to pinn_$(name).jld2"