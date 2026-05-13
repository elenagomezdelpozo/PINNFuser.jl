include("../src/Lib.jl")

include("../src/Parameters.jl")
using .ParametersMod: parameters

name = parameters.name # model to be visualized

LibInfuser.Plots_model("pinn_$(name)_local") 
