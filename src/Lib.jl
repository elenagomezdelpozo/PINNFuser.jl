module LibInfuser

include("PINN_Infuser.jl")
include("Losses.jl")
include("Model.jl")
include("Saving_plots.jl")

# module names
export PINNInfuser_module
export Losses_module
export Model_module
export Savingplots_module


using .LibInfuser: PINNInfuser_module
using .LibInfuser.PINNInfuser_module: PINN_Infuser_funct

using .Losses_module
using .Model_module
using .Savingplots_module

end