include("../src/Lib.jl")
using .LibInfuser

include("../src/Parameters.jl")
using .ParametersMod: parameters

name = parameters.name # name of the training to test
@info "Testing model: $(name)"  

training_ode_parameters, odes = LibInfuser.generate_patients(parameters.number_of_patients; seed = parameters.seed)  # all patients used in training
LibInfuser.Tester_f("pinn_training_$(name)", training_ode_parameters; test = false) 

new_ode_parameters, odes = LibInfuser.generate_patients(1; seed = 1)  # generate a single patient and extract its parameters
LibInfuser.Tester_f("pinn_test_$(name)", new_ode_parameters; test = true) 