include("../src/Lib.jl")
using .LibInfuser

include("../src/Parameters.jl")
using .ParametersMod: parameters

name = parameters.name # model to be visualized
i = parameters.i

LibInfuser.Plots_f("pinn_model_$(name)", i) 
