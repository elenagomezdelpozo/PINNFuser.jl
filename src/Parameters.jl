module ParametersMod

using DelimitedFiles
using StaticArrays

working_on = "local" # CHANGE "hpc" or "local"

if working_on == "hpc"
    i = parse(Int, ARGS[1])
    config_type = length(ARGS) > 1 ? ARGS[2] : "single_data"
elseif working_on == "local"
    name = "all_patients_data"
    active = ["data"] # "data, physics, mass, zero_mean, negativity, firstderiv, periodicity" 
    i = 7 # 1:6 for single variable, 7 for all variables
    early_stopping_start = 20
end

training = (
    vars = [1,2,3,4,5,6],
    n_neurons_per_layer = 10,
    lr = 1e-3,
    dtmax = 1e-2,
    nn_output_weight = 1.0,
    iterations = 1000,
    initialisation = 1e-3,
    plot_every = 1,
    num_of_cycles = 1,
)

changeable = (
    name = name,
    active = active,
    i = i,
    early_stopping_start = early_stopping_start,
    plot_time = (23.0, 30.0)
)

independent = (
    tspan = (0.0, 20.0),
    num_of_samples_per_cycle = 150,
    τ = 1.0,
    Eshift = 0.0,

        # pLV, pLA, psa, psv, Vlv, Vla, Qav, Qmv, Qs, Qsv 
    u0 =  [8.0, 8.0, 30.0, 21.5, 130.0, 75.0, 0.0, 0.0, 0.0, 0.0],

              # τ, τₑₛ_lv, τₑₚ_lv, τₑₛ_la, τₑₚ_la 
    t_params = [1.0, 0.3, 0.45, 0.92, 0.09],  

                # Rmv, Zao, Rs, Rsv, Csa, Csv, Eₘₐₓ_lv, Eₘᵢₙ_lv, Eₘₐₓ_la, Eₘᵢₙ_la 
    ode_params = [0.013, 0.002, 1.292, 0.07, 1.023, 10.9, 5.2, 0.0709, 0.2, 0.06],

    extrapolation_tspan = (0.0, 30.0),

    number_of_patients = 10

)

Rmv, Zao, Rs, Rsv, Csa, Csv, Emax_lv, Emin_lv, Emax_la, Emin_la = independent.ode_params

if working_on == "hpc"
    loaded_data = readdlm("/net/people/plgrid/plgelenagdelpozo/CV_0D_models/PINNFuser.jl/data/target_data.txt")
    plotting = false
    num_of_samples = Int(independent.num_of_samples_per_cycle * training.num_of_cycles)
    tsteps = range(independent.extrapolation_tspan[2]-independent.τ * independent.num_of_cycles, independent.extrapolation_tspan[2], length = num_of_samples) # for training
elseif working_on == "local"
    loaded_data = readdlm("data/target_data.txt")
    plotting = true
    num_of_samples = Int(independent.num_of_samples_per_cycle * (independent.tspan[2] - independent.tspan[1])) 
    tsteps = range(independent.tspan[1], independent.tspan[2], length = num_of_samples)
end

extrap_original_data = Array{Float64}(loaded_data)[
    1:Int(floor(independent.extrapolation_tspan[2] * independent.num_of_samples_per_cycle)), :
]
original_data = extrap_original_data[1001:1000 + num_of_samples, :]

config = (
    data_vars = [1, 2, 3, 4, 5, 6],
    data_weight = 1e-2,
    physics_vars = [1, 2, 3],
    physics_weight = 1e-5,
    mass_conservation_weight = 1.0,
    zm_vars = [1, 2, 3, 4, 5, 6],
    zm_weight = 1e-5,
    neg_vars = [5, 6],
    neg_weight = 10.0,
    firstderiv_vars = [1, 2, 3, 4, 5, 6],
    deriv_weight = 1e-5,
    periodic_vars = [1,2,3,4,5,6],
    periodic_weight = 1e-5
)
plot_params = (
    labels = [
        "pLV",
        "pLA",
        "psa",
        "psv",
        "Vlv",
        "Vla",
        "Qav",
        "Qmv",
        "Qs",
        "Qsv"
    ],
    ylims = [
        (0, 130),
        (4, 8),
        (60, 140),
        (22,24),
        (30, 120),
        (30, 70)
    ]
)
dependent = (
    plotting = plotting,
    num_of_samples = num_of_samples,
    tsteps = tsteps,
    Rmv = Rmv,
    Zao = Zao,
    Rs = Rs,
    Rsv = Rsv,
    Csa = Csa,
    Csv = Csv,
    Emax_lv = Emax_lv,
    Emin_lv = Emin_lv,
    Emax_la = Emax_la,
    Emin_la = Emin_la,

    loaded_data = loaded_data,
    extrap_original_data = extrap_original_data,
    original_data = original_data,

    config = config
)

parameters = merge((working_on = working_on,), changeable, plot_params, independent, dependent, training)

end # module