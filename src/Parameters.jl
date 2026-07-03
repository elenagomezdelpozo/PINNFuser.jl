__precompile__(false)  # Add this here
module ParametersMod

using DelimitedFiles
using Lux

working_on = "hpc" # CHANGE "hpc" or "local"

i = parse(Int, ARGS[1])
# i = 3

number_of_patients = 50 # CHANGE number of patients (20, 50, 100, 200, 500, 1000)

actives = ["data", "physics", "mass", "zero_mean", "negativity", "periodicity"]
names = [
    "data",             #1
    "data&physics",     #2
    "data&mass",        #3
    "data&zero_mean",   #4
    "data&negativity",  #5
    "data&periodicity"  #6
]

active = try
    i == 1 ? [actives[1]] : [actives[1], actives[i]]
catch e
    println("Error because of i definition: $e")
    nothing  # or some fallback value
end

changeable = ( 
    working_on = working_on,
    name = names[i]
)

if working_on == "hpc"
    data_dir = "/net/afscra/people/plgelenagdelpozo/PINNFuser.jl/data/Model/data"
    plotting = false
    savepath = "/net/afscra/people/plgelenagdelpozo/PINNFuser.jl/trainings/pinn_$(number_of_patients)_$(changeable.name)_$(working_on).jld2"

elseif working_on == "local"
    data_dir = "data/Model/data"
    plotting = true
    savepath = "trainings/pinn_$(number_of_patients)_$(changeable.name)_hpc.jld2"
end

# Load all patients into a list
all_patient_files = sort(filter(f -> startswith(f, "patient_") && endswith(f, ".csv"),
                                readdir(data_dir)))
all_patient_files = all_patient_files[1:100]   # take all
training_patient_files = all_patient_files[1:number_of_patients]  # take first N patients

loaded_all_data_list = map(all_patient_files) do fname
    readdlm(joinpath(data_dir, fname), ',', Float64, '\n'; skipstart=1)
end
loaded_training_data_list = map(training_patient_files) do fname
    readdlm(joinpath(data_dir, fname), ',', Float64, '\n'; skipstart=1)
end

training = (
    vars = [1,2,3,4,5,6],
    n_neurons_per_layer = 10,
    lr = 1e-3,
    dtmax = 1e-2,
    nn_output_weight = 1.0, # possibly lower
    iterations = 1000,
    initialisation = 1e-3,
    plot_every = 1,
    num_of_cycles = 1,
    seed = 42,
    active = active,
    early_stopping_start = 20,
    range_to_plot = 5.0,
    number_of_patients = number_of_patients
)

# Do not change these below
independent = (
    tspan = (0.0, 30.0),
    num_of_samples_per_cycle = 150,
    τ = 1.0,
    Eshift = 0.0,
         #  pLV,  pLA,  psa,  psv,  Vlv,   Vla, Qav, Qmv, Qs, Qsv 
    u0 =  [11.0, 11.0, 40.0, 6.0, 200.0, 115.0, 0.0, 0.0, 0.0, 0.0],
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

function process_patient_data(loaded_data, num_cycles)
    extrap = Array{Float64}(loaded_data)[
        1:Int(floor(independent.extrapolation_tspan[2] * independent.num_of_samples_per_cycle)), :
    ]
    original = extrap[1051:1050 + num_cycles * independent.num_of_samples_per_cycle, :]
    return (extrap_original_data = extrap, original_data = original)
end

processed_training_list = map(process_patient_data, loaded_training_data_list, [training.num_of_cycles for _ in 1:length(loaded_training_data_list)])
original_training_data_list = [p.original_data for p in processed_training_list]   # this goes into PINN_Infuser_f

processed_all_data_list = map(process_patient_data, loaded_all_data_list, [Int(training.range_to_plot) for _ in 1:length(loaded_all_data_list)])
original_all_data_list = [p.original_data for p in processed_all_data_list]   # this goes into PINN_Infuser_f

plot_time = range(independent.tspan[2]-training.range_to_plot, independent.tspan[2], length = Int(independent.num_of_samples_per_cycle * (training.range_to_plot)))
training_time = range(independent.tspan[2]- independent.τ*training.num_of_cycles, independent.tspan[2], length = Int(training.num_of_cycles * independent.num_of_samples_per_cycle))

config = (
    data_vars = [1, 2, 3, 4, 5, 6],
    data_weight = 1e-1,
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
    min_loss = 1e-7,
    dt = dt
)

NN = Lux.Chain(
    Lux.Dense(2 * length(independent.u0), training.n_neurons_per_layer, tanh),
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
    data_dir = data_dir,
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
    loaded_training_data_list       = loaded_training_data_list,       # raw, if needed elsewhere
    original_training_data_list     = original_training_data_list,     # → goes into PINN_Infuser_f
    loaded_all_data_list            = loaded_all_data_list,            # raw, if needed elsewhere
    original_all_data_list          = original_all_data_list,          # → used in Tester
    config = config
)

parameters = merge(config, changeable, plot_params, independent, dependent, training)

end # module