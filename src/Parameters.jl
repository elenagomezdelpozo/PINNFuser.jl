__precompile__(false)  # Add this here
module ParametersMod

using DelimitedFiles
using Lux

working_on = "hpc" # CHANGE "hpc" or "local"

i = parse(Int, ARGS[1])
# i = 4

number_of_patients = 30 # CHANGE number of patients (20, 50, 100, 200, 500, 1000)

actives = ["data", "physics", "mass", "zero_mean", "negativity", "firstderiv", "periodicity"]
names = [
    "data",             #1
    "data&physics",     #2
    "data&mass",        #3
    "data&zero_mean",   #4
    "data&negativity",  #5
    "data&periodicity"  #6
]

active = i == 1 ? [actives[1]] : [actives[1], actives[i]]

changeable = ( 
    working_on = working_on,
    name = names[i],
    active = active,
    early_stopping_start = 100,
    range_to_plot = 5.0,
    number_of_patients = number_of_patients
)

if working_on == "hpc"
    loaded_data = readdlm("/net/afscra/people/plgelenagdelpozo/PINNFuser.jl/data/target_data.txt")
    plotting = false
    savepath = "/net/afscra/people/plgelenagdelpozo/PINNFuser.jl/trainings/pinn_$(number_of_patients)_$(changeable.name)_$(working_on).jld2"

elseif working_on == "local"
    loaded_data = readdlm("data/target_data.txt")
    plotting = true
    savepath = "trainings/pinn_$(number_of_patients)_$(changeable.name)_hpc.jld2"
end

training = (
    vars = [1,2,3,4,5,6],
    n_neurons_per_layer = 10,
    lr = 1e-4,
    dtmax = 1e-2,
    nn_output_weight = 1.0, # possibly lower
    iterations = 1000,
    initialisation = 1e-3,
    plot_every = 1,
    num_of_cycles = 1,
    seed = 42
)

# Do not change these below
independent = (
    tspan = (0.0, 30.0),
    num_of_samples_per_cycle = 150,
    τ = 1.0,
    Eshift = 0.0,
        #  pLV, pLA, psa,  psv,  Vlv,   Vla, Qav, Qmv, Qs, Qsv 
    u0 =  [9.0, 9.0, 70.0, 21.0, 120.0, 55.0, 0.0, 0.0, 0.0, 0.0],
              # τ, τₑₛ_lv, τₑₚ_lv, τₑₛ_la, τₑₚ_la 
    t_params = [1.0, 0.3, 0.45, 0.92, 0.09],  
                # Rmv, Zao, Rs, Rsv, Csa, Csv, Eₘₐₓ_lv, Eₘᵢₙ_lv, Eₘₐₓ_la, Eₘᵢₙ_la 
    ode_params = [0.013, 0.002, 1.292, 0.07, 1.023, 10.9, 5.2, 0.0709, 0.2, 0.06],
    extrapolation_tspan = (0.0, 40.0)
)

Rmv, Zao, Rs, Rsv, Csa, Csv, Emax_lv, Emin_lv, Emax_la, Emin_la = independent.ode_params

num_of_samples_tsteps = Int(independent.num_of_samples_per_cycle * (independent.tspan[2] - independent.tspan[1])) 
tsteps = range(independent.tspan[1], independent.tspan[2], length = num_of_samples_tsteps)
dt = tsteps[2] - tsteps[1]
extrap_original_data = Array{Float64}(loaded_data)[
    1:Int(floor(independent.extrapolation_tspan[2] * independent.num_of_samples_per_cycle)), :
]
original_data = extrap_original_data[1001:1000 + training.num_of_cycles * independent.num_of_samples_per_cycle, :]
plot_time = range(independent.tspan[2]-changeable.range_to_plot, independent.tspan[2], length = Int(independent.num_of_samples_per_cycle * (changeable.range_to_plot)))
training_time = range(independent.tspan[2]- independent.τ*training.num_of_cycles, independent.tspan[2], length = Int(training.num_of_cycles * independent.num_of_samples_per_cycle))

config = (
    data_vars = [1, 2, 3, 4, 5, 6],
    data_weight = 1e-2,
    physics_vars = [1, 2, 3],
    physics_weight = 1e-5,
    mass_conservation_weight = 1.0,
    zm_vars = training.vars,
    zm_weight = 1e-5,
    neg_vars = [5, 6],
    neg_weight = 10.0,
    firstderiv_vars = [1, 2, 3, 4, 5, 6],
    deriv_weight = 1e-5,
    periodic_vars = [1,2,3,4,5,6],
    periodic_weight = 1e-5,
    min_loss = 1e-4,
    dt = dt
)

NN = Lux.Chain(
    Lux.Dense(length(independent.u0), training.n_neurons_per_layer, tanh),
    Lux.Dense(training.n_neurons_per_layer, training.n_neurons_per_layer, tanh),
    Lux.Dense(training.n_neurons_per_layer, training.n_neurons_per_layer, tanh),
    Lux.Dense(training.n_neurons_per_layer, training.n_neurons_per_layer, tanh),
    Lux.Dense(training.n_neurons_per_layer, length(training.vars)),
)  

plot_params = (
    labels = [
        "Plv",
        "Pla",
        "Psa",
        "Psv",
        "Vlv",
        "Vla",
        "Qav",
        "Qmv",
        "Qs",
        "Qsv"
    ],
    ylims = [
        (0, 150),
        (4, 10),
        (40, 140),
        (21,28),
        (0, 120),
        (30, 90)
    ],
    units = [
        "mmHg",
        "mmHg",
        "mmHg",
        "mmHg",
        "mL",
        "mL",
        "mL/s",
        "mL/s",
        "mL/s",
        "mL/s"
    ]
)
dependent = (
    NN = NN,
    plotting = plotting,
    savepath = savepath,
    num_of_samples_tsteps = num_of_samples_tsteps,
    tsteps = tsteps,
    plot_time = plot_time,
    training_time = training_time,
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

parameters = merge(config, changeable, plot_params, independent, dependent, training)

end # module