module PINNExtrapolator_new

using Lux, DelimitedFiles, OrdinaryDiffEq, Statistics, LinearAlgebra

export PINN_Extrapolator_new

"""

Solves a PINN-infused ODE problem to extrapolate its trajectory and saves the result to a file.

This function takes a pre-trained neural network's parameters (`pretrained_params`) and an `ODEProblem` that uses them. It then solves this problem over a new time span `tspan` starting from a new initial condition `u0`, generating a specified number of sample points. The resulting time series data is saved to `path_to_save`.

# Arguments
- `base_problem::SciMLBase.ODEProblem`: Base ODE problem.
- `nn::Lux.Chain`: The Lux neural network model structure.
- `pretrained_params::Tuple{Any, Any}`: The trained parameters of the neural network.
- `tspan::Tuple{Float64, Float64}`: The time interval `(t_start, t_end)` for the extrapolation.
- `num_of_samples::Int`: The number of evenly spaced time points to generate and save within the `tspan`.
- `path_to_save::String`: The full file path (e.g., `"data/prediction.csv"`) where the output will be saved.
- `nn_output_weight::Float64 = 0.1`: The weight factor for the NN infusion in ODE.
- `reltol::Float64 = 1e-6`: The relative tolerance for the ODE solver.
- `abstol::Float64 = 1e-6`: The absolute tolerance for the ODE solver.
- `dtmax::Float64 = 1e-2`: The maximum time step for the ODE solver.
"""
function PINN_Extrapolator_new(
    ode_problem::SciMLBase.ODEProblem,
    nn::Lux.Chain,
    pretrained_params::Tuple{Any,Any, Any, Any},
    tspan::Tuple{Float64,Float64},
    num_of_samples::Int,
    path_to_save::String;
    nn_output_weight::Float64 = 0.1,
    reltol::Float64 = 1e-6,
    abstol::Float64 = 1e-6,
    dtmax::Float64 = Inf,
    nn_vars::Union{Nothing,Vector{Int}} = nothing,
    τ::Float64 = 1.0,
    clip_nn::Float64 = 5.0,
)::Nothing

    trained_p, trained_st, U_MEAN, U_STD = pretrained_params
    new_tseps = range(tspan[1], tspan[2], length = num_of_samples)

    if nn_vars === nothing
        nn_vars = collect(1:length(ode_problem.u0))
    end    

    function pinn_ode!(du, u, p, t)
        ode_problem.f(du, u, ode_problem.p, t) # original ODE problem
        nn_output = nn(u, trained_p, trained_st)[1] # NN correction
        for i in nn_vars
            du[i] += nn_output_weight * nn_output[i]
        end
    end

    pinn_problem = ODEProblem(pinn_ode!, ode_problem.u0, tspan)

    solved_pinn = solve(
        pinn_problem,
        Vern7(),
        saveat = new_tseps,
        reltol = reltol,
        abstol = abstol,
        dtmax = dtmax,
    )

    pred_mat = hcat(solved_pinn.u...)'

    writedlm(path_to_save, pred_mat, ',')
end

end # module PINNExtrapolator_new