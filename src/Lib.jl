__precompile__(false)  # Add this here
module LibInfuser

include("PINN_Infuser.jl")
include("Losses.jl")
include("Model.jl")
include("Results_visualisation.jl")

# module names
export PINNInfuserMod
export LossesMod
export ModelMod
export SavingplotsMod


using .LibInfuser: PINNInfuserMod
using .LibInfuser.PINNInfuserMod: PINN_Infuser_f

using .LossesMod

using .ModelMod
using .ModelMod: NIK_2ch!

using .PlotsMod
using .LibInfuser.PlotsMod: Plots_f

end