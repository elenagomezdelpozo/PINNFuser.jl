include("../src/Lib.jl")
using .LibInfuser

include("../src/Parameters.jl")
using .ParametersMod: parameters

name = parameters.name # name of the training to test
new_ode_parameters = LibInfuser.generate_patients(10; seed = 42)[9, :]  # generate a single patient and extract its parameters
@info "New ODE parameters for testing: $(new_ode_parameters)"
LibInfuser.Tester_f("pinn_model_$(name)", new_ode_parameters) 