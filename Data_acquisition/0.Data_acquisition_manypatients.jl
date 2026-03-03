# 0.Data_acquisiton
using CellMLToolkit, ModelingToolkit, OrdinaryDiffEq
using DelimitedFiles, StableRNGs, Statistics, Plots
using Plots

number_of_patients = 10
# 1. Setup Model and Problem
ml = CellModel("/Applications/Desktop/CODE/PINNFuser.jl/Data_acquisition/ModelMain.cellml")
sys = ml.sys
tspan = (0.0, 60.0)
prob = ODEProblem(ml, tspan) 
tsteps = range(tspan[1], tspan[2], length = Int(tspan[2] * 150))

# 2. Map the symbols using the Dictionary approach
all_params = parameters(sys)
p_lookup = Dict(string(p) => p for p in all_params)

param_symbols = [
    p_lookup["ParaLHeart₊CVmi"],   # Mitral valve coefficient
    p_lookup["ParaLHeart₊CVao"],   # Aortic valve coefficient
    p_lookup["ParaSys₊Rscp"],      # Systemic capillary resistance
    p_lookup["ParaSys₊Rsvn"],      # Venous resistance
    p_lookup["ParaSys₊Csvn"],      # Venous compliance
    p_lookup["ParaSys₊Csat"],      # Arterial compliance
    p_lookup["ParaLHeart₊ElvMax"], # LV max elastance
    p_lookup["ParaLHeart₊ElvMin"], # LV min elastance
    p_lookup["ParaLHeart₊ElaMax"], # LA max elastance
    p_lookup["ParaLHeart₊ElaMin"]  # LA min elastance
]

# 3. Patient Generation Logic (R_sys/R_svn ratio)
function generate_patients(n_samples::Int)
    patients = Matrix{Float64}(undef, n_samples, 10)
    loguniform(a, b) = exp(rand() * (log(b) - log(a)) + log(a))

    for i in 1:n_samples
        CV_mi = rand(100:100:1000)
        CV_ao = rand(200:200:2000)
        R_sys = rand() * 2.0 
        ratio_sys = rand(5.0:0.5:20.0)
        R_svn = clamp((R_sys + 0.55) / ratio_sys, 0.05, 0.5)
        C_sas = loguniform(0.5, 2.5)
        ratio_comp = rand(5.0:0.5:25.0)
        C_svn = ratio_comp * C_sas
        Elvmin = loguniform(0.03, 0.15)
        ratio_lv = rand(10.0:1.0:40.0)
        Elvmax = ratio_lv * Elvmin
        Elamin = loguniform(0.05, 0.2)
        ratio_la = rand(3.0:1.0:15.0)
        Elamax = ratio_la * Elamin
        if Elvmax <= Elamax; Elvmax = 1.2 * Elamax; end

        patients[i, :] = [CV_mi, CV_ao, R_sys, R_svn, C_svn, C_sas, Elvmax, Elvmin, Elamax, Elamin]
    end
    return patients
end

# these are CV_mi, CV_ao, R_sys, R_svn, C_svn, C_sas, Elvmax, Elvmin, Elamax, Elamin as defined by the model online
original_params = [400.0, 350.0, 0.52, 0.075, 20.5, 0.08, 2.5, 0.1, 0.25, 0.15] 
indices = [44, 43, 99, 100, 101, 89, 26, 27, 22, 23] # indices of the params in prob.p[1][index]

function simulate_patient(prob, p_values, tsteps)
    new_p = copy(prob.p)
    for i in 1:length(p_values)
        new_p[1][indices[i]] = p_values[i]
    end
    println("Simulating patient with parameters: ", p_values)
    new_prob = remake(prob, p = new_p)
    return solve(new_prob, Tsit5(); saveat = tsteps, reltol = 1e-4, abstol = 1e-7)
end
sol = simulate_patient(prob, all_patients[1, :], tsteps)
# Parameters for all patients
all_patients = generate_patients(number_of_patients)

# Plotting
labels = [
    "pLV",  # Left ventricle pressure
    "pLA",  # Left atrium pressure
    "pSA",  # Systemic arterial pressure
    "pSV",  # Systemic venous pressure
    "VLV",  # LV volume
    "VLA",  # LA volume
    "Qav",  # Aortic valve flow
    "Qmv",  # Mitral valve flow
    "Qs",   # Systemic flow
    "Qsv"   # Venous flow
]

all_patient_data = []

for patient_idx in 1:number_of_patients
    sol = simulate_patient(prob, all_patients[patient_idx, :], tsteps)
    data_patient = hcat(
        sol[sys.LV.Pi],  # pLV
        sol[sys.LA.Pi],  # pLA
        sol[sys.Sas.Pi], # pSA
        sol[sys.Svn.Po], # pSV
        sol[sys.LV.V],   # VLV
        sol[sys.LA.V],   # VLA
        sol[sys.LV.Qo],  # Qav
        sol[sys.LA.Qo],  # Qmv
        sol[sys.Sat.Qi], # Qs
        sol[sys.Svn.Qi]  # Qsv
    )
    push!(all_patient_data, data_patient)
end

tsteps_plot = tsteps[8000:9000]  # Example: plot last cycles
plots = [
    begin
        p = plot(title = labels[output_idx], xlabel="Time [s]", ylabel=labels[output_idx], lw=2)
        for patient_idx in 1:number_of_patients
            plot!(p, tsteps_plot, all_patient_data[patient_idx][8000:9000, output_idx], label="Patient $patient_idx")
        end
        p
    end
    for output_idx in 1:10
]

plot(plots..., layout = (5,2), size=(1000, 800))

"""
# For all patients
for i in 1:size(all_patients, 1)
    sol = simulate_patient(prob, all_patients[i, :], param_symbols, tsteps)
    filename = "/Applications/Desktop/CODE/Data_acquisition/data/patient_" * string(i) * ".txt"
    writedlm(filename, sol.u)
end
"""
