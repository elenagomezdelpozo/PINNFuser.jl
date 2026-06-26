__precompile__(false)  # Add this here
module LibInfuser

include("PINN_Infuser.jl")
include("Losses.jl")
include("Model.jl")
include("Visualization.jl")
include("Tester.jl")
include("Generate_patients.jl")

# module names
export PINNInfuserMod
export LossesMod
export ModelMod
export SavingplotsMod
export TestMod
export ODEPlotsMod
export PatientsMod

using .PINNInfuserMod: PINN_Infuser_f

using .LossesMod

using .ModelMod: NIK_2ch, NIK_2ch!

using .PlotsMod: Plot_ODE, Plot_all_patients, Plot_target

using .TestMod: Tester_f

using .PatientsMod: generate_patients, save_params_manifest, simulate_patient, save_patient_data

end