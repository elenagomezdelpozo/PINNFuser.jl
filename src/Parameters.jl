__precompile__(false) 
module ParametersMod

using DelimitedFiles

config_type = get(ENV, "CONFIG_TYPE", "all")

if config_type == "single"
    include("Configs_single_vars.jl")
else
    include("Configs_all_vars.jl")
end
using .ConfigsMod: configs

export parameters

i = parse(Int, ARGS[1])

changeable = (

    working_on = "hpc", # "hpc" or "local"

    name = configs[i].name,          
    active = configs[i].active,
    early_stopping_start = configs[i].early_stopping_start,

    vars = [1,2,3,4,5,6],
    nn_vars = [1],
    n_neurons_per_layer = 10,
    lr = 1e-4,
    dtmax = 1e-2,
    nn_output_weight = 1.0,
    iterations = 1000,
    initialisation = 1e-3,
    plot_every = 1,
    plotting = false
)

independent = (
    tspan = (0.0, 30.0),
    num_of_samples_per_cycle = 150,
    num_of_cycles = 1,

    τ = 1.0,
    Eshift = 0.0,

        # pLV, pLA, psa, psv, Vlv, Vla, Qav, Qmv, Qs, Qsv 
    u0 = [8.0, 8.0, 30.0, 21.5, 130.0, 75.0, 0.0, 0.0, 0.0, 0.0],

              # τ, τₑₛ_lv, τₑₚ_lv, τₑₛ_la, τₑₚ_la 
    t_params = [1.0, 0.3, 0.45, 0.92, 0.09],  

                # Rmv, Zao, Rs, Rsv, Csa, Csv, Eₘₐₓ_lv, Eₘᵢₙ_lv, Eₘₐₓ_la, Eₘᵢₙ_la 
    ode_params = [0.013, 0.002, 1.292, 0.07, 1.023, 10.9, 5.2, 0.0709, 0.2, 0.06],

    extrapolation_tspan = (0.0, 30.0),

)

num_of_samples = independent.num_of_samples_per_cycle * independent.num_of_cycles

tsteps = range(29.0, 30.0, length = num_of_samples) # for training

Rmv, Zao, Rs, Rsv, Csa, Csv, Emax_lv, Emin_lv, Emax_la, Emin_la = independent.ode_params

if changeable.working_on == "hpc"
    loaded_data = readdlm("/net/people/plgrid/plgelenagdelpozo/CV_0D_models/PINNFuser.jl/data/original_data_2Ch.txt")
elseif working_on == "local"
    changeable.loaded_data = readdlm("../data/original_data_2Ch.txt")
end

extrap_original_data = Array{Float64}(loaded_data)[
    1:Int(floor(independent.extrapolation_tspan[2] * independent.num_of_samples_per_cycle)), :
]
original_data = extrap_original_data[1501:1500 + num_of_samples, :]

config = (
    data_vars = [1, 2, 3, 4, 5, 6],
    data_weight = 1.0,
    physics_vars = [1, 2, 3],
    physics_weight = 1.0,
    mass_conservation_weight = 1.0,
    zm_vars = [1, 2, 3, 4, 5, 6],
    zm_weight = 1.0,
    neg_vars = [5, 6],
    neg_weight = 10.0,
    firstderiv_vars = [1, 2, 3, 4, 5, 6],
    deriv_weight = 1e-4,
    periodic_vars = [1,2,3,4,5,6],
    periodic_weight = 1.0
)
plot_params = (
    plot_time = (0,7),

    labels = [
        "pLV",
        "pLA",
        "psa",
        "psv",
        "Vlv",
        "Vla"
    ],
    ylims = [
        (0, 130),
        (4, 10),
        (40, 140),
        (22,24),
        (30, 150),
        (20, 70)
    ]
)
dependent = (
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

parameters = merge(changeable, plot_params, independent, dependent)

end # module