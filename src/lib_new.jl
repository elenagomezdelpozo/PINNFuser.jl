module LibInfuserNew

include("../src/PINN_Parameter_Tuner.jl")
include("../src/PINN_Infuser_new.jl")
include("../src/PINN_Extrapolator_new.jl")
include("../src/PINN_Symbolic_Regressor.jl")
include("../src/PINN_Plotter_new.jl")
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
