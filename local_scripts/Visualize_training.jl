include("../src/Lib.jl")

include("../src/Parameters.jl")
using .ParametersMod: parameters

name = parameters.name # model to be visualized

LibInfuser.Plot_model("pinn_$(name)_$(parameters.working_on)") 
