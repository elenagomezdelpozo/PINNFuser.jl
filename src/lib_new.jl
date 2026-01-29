module LibInfuserNew

include("/Applications/Desktop/CODE/Thesis/PINNFuser.jl/src/PINN_Parameter_Tuner.jl")
include("/Applications/Desktop/CODE/Thesis/PINNFuser.jl/src/PINN_Infuser_new.jl")
include("/Applications/Desktop/CODE/Thesis/PINNFuser.jl/src/PINN_Extrapolator_new.jl")
include("/Applications/Desktop/CODE/Thesis/PINNFuser.jl/src/PINN_Symbolic_Regressor.jl")
include("/Applications/Desktop/CODE/Thesis/PINNFuser.jl/src/PINN_Plotter_new.jl")
export PINNParamTuner
export PINNInfuser_new
export PINNExtrapolator_new
export PINNSymbolicRegressor
export PINNPlotter_new

using .PINNParamTuner
using .PINNInfuser_new
using .PINNExtrapolator_new
using .PINNSymbolicRegressor
using .PINNPlotter_new

end
