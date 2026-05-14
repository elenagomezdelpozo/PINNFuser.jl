__precompile__(false)  # Add this here
module LibInfuser

include("PINN_Infuser.jl")
include("Losses.jl")
include("Model.jl")
include("Results_visualization.jl")
include("Tester.jl")

# module names
export PINNInfuserMod
export LossesMod
export ModelMod
export SavingplotsMod
export TestMod
export ODEPlotsMod

using .PINNInfuserMod: PINN_Infuser_f

using .LossesMod

using .ModelMod: NIK_2ch, NIK_2ch!

using .PlotsMod: Plot_model, Plot_ODE, Plot_all_patients, Plot_target

using .TestMod: Tester_f, generate_patients

end