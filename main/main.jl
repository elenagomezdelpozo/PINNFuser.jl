using OptimizationOptimisers: ADAM
using JLD2

include("../src/Lib.jl")
using .LibInfuser

include("../src/Parameters.jl")
using .ParametersMod: parameters

patients_params, patients_odes = LibInfuser.generate_patients(parameters.number_of_patients, seed = 42)
name = parameters.name

nn_vars = parameters.vars
NN = parameters.NN

@info "Starting training for $(parameters.name) with $(parameters.number_of_patients) patients on $(parameters.working_on)..."
trained_p, trained_st, losses = LibInfuser.PINN_Infuser_f(
    patients_odes,
    patients_params,
    NN,
    parameters.original_data;
    nn_vars = nn_vars,
    optimizer = ADAM,
    reltol = Float32(1e-6),
    abstol = Float32(1e-6),
    early_stopping = true,
    plotting = parameters.plotting
)

jldsave(parameters.savepath;
        trained_p = trained_p,
        trained_st = trained_st,
        losses = losses
        )

@info "Model saved successfully to pinn_$(name).jld2"