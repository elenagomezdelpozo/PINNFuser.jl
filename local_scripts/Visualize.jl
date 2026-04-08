include("../src/Lib.jl")
using .LibInfuser

include("../src/Parameters.jl")
using .ParametersMod: parameters

name = parameters.name
i = parameters.i

LibInfuser.Plots_f("pinn_model_$(name)_$(i).jld2", i)