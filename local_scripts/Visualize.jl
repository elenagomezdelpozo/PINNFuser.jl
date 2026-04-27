include("../src/Lib.jl")
using .LibInfuser

include("../src/Parameters.jl")
using .ParametersMod: parameters

name = parameters.name

for i in 1:6
    LibInfuser.Plots_f("pinn_model_$(name)_$(i)", i)
end

name = parameters.name * "_all"

LibInfuser.Plots_f("pinn_model_$(name)", 7) # use 7 for all variables


