include("../src/Lib.jl")
using .LibInfuser

include("../src/Parameters.jl")
using .ParametersMod: parameters

name = parameters.name # name of the training to test

training_ode_parameters, odes = LibInfuser.generate_patients(parameters.number_of_patients; seed = parameters.seed)  # all patients used in training
@info "ODE parameters used for training: $(training_ode_parameters)"
LibInfuser.Tester_f("pinn_training_$(name)", training_ode_parameters; loss = true) 

new_ode_parameters, odes = LibInfuser.generate_patients(1; seed = 1)  # generate a single patient and extract its parameters
@info "New ODE parameters for testing: $(new_ode_parameters)"
LibInfuser.Tester_f("pinn_test_$(name)", new_ode_parameters; loss = false) 