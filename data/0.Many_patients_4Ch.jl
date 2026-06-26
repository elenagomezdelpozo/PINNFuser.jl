# 0. Data_acquisition
using CellMLToolkit, ModelingToolkit, OrdinaryDiffEq
using DelimitedFiles, Logging, Plots, EzXML
using Printf, Random

include("../src/Parameters.jl")
using .ParametersMod: parameters

include("../src/Lib.jl")
using .LibInfuser
# =============================================================================
# 5. MAIN LOOP
# =============================================================================
using Printf, Random

# Create output directory
mkpath(DATA_PATH)

# Make backups of CellML parameter files (restored at end)
cp(HEART_FILE, HEART_FILE * ".bak", force = true)
cp(SYS_FILE,   SYS_FILE   * ".bak", force = true)

# Generate patients
all_patients = LibInfuser.generate_patients(NUMBER_OF_PATIENTS)

# Save parameter manifest upfront — safe even if simulation crashes midway
LibInfuser.save_params_manifest(all_patients)

# Simulate and save each patient
all_patient_data = []
t_start = time()

for patient_idx in 1:NUMBER_OF_PATIENTS
    print("Simulating patient $patient_idx / $NUMBER_OF_PATIENTS ... ")
    t0 = time()

    data = LibInfuser.simulate_patient(all_patients[patient_idx, :], tsteps)
    push!(all_patient_data, data)
    LibInfuser.save_patient_data(patient_idx, data, all_patients[patient_idx, :])

    elapsed = round(time() - t0, digits = 1)
    remaining = round((time() - t_start) / patient_idx * (NUMBER_OF_PATIENTS - patient_idx), digits = 0)
    println("done ($(elapsed)s) | est. remaining: $(remaining)s")
end

# Restore original CellML files
cp(HEART_FILE * ".bak", HEART_FILE, force = true)
cp(SYS_FILE   * ".bak", SYS_FILE,   force = true)
println("\nOriginal CellML files restored.")
println("All data saved to: $DATA_PATH")

# =============================================================================
# 6. PLOT (last ~7 seconds = steady-state cycles)
# =============================================================================
tsteps_vec  = collect(tsteps)
tsteps_plot = tsteps_vec[8000:9000]

plots_list = [
    begin
        p = plot(title = LABELS[output_idx], xlabel = "Time [s]",
                 ylabel = LABELS[output_idx], lw = 2, legend = :outertopright)
        for patient_idx in 1:NUMBER_OF_PATIENTS
            plot!(p, tsteps_plot, all_patient_data[patient_idx][8000:9000, output_idx],
                  label = "P$patient_idx")
        end
        p
    end
    for output_idx in 1:10
]

fig = plot(plots_list..., layout = (5, 2), size = (1200, 1000))
savefig(fig, joinpath(DATA_PATH, "all_patients_overview.png"))
display(fig)
println("Plot saved → $(joinpath(DATA_PATH, "all_patients_overview.png"))")