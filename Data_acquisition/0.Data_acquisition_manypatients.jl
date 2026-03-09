# 0. Data_acquisition
using CellMLToolkit, ModelingToolkit, OrdinaryDiffEq
using DelimitedFiles, Logging, Plots, EzXML
using Printf, Random

# =============================================================================
# CONFIGURATION
# =============================================================================
const NUMBER_OF_PATIENTS = 100
const BASE_PATH  = "/Applications/Desktop/CODE/PINNFuser.jl/Data_acquisition"
const DATA_PATH  = joinpath(BASE_PATH, "data")
const HEART_FILE = joinpath(BASE_PATH, "ParaHeart.cellml")
const SYS_FILE   = joinpath(BASE_PATH, "ParaSys.cellml")
const tspan      = (0.0, 60.0)
const tsteps     = range(tspan[1], tspan[2], length = Int(tspan[2] * 150))

# Output labels (for reference and plot titles)
const LABELS = ["pLV", "pLA", "pSA", "pSV", "VLV", "VLA", "Qav", "Qmv", "Qs", "Qsv"]

# =============================================================================
# 1. PATIENT GENERATION
# =============================================================================
# Parameters: [CV_mi, CV_ao, R_scp, R_svn, C_svn, C_sas, Elvmax, Elvmin, Elamax, Elamin]
function generate_patients(n_samples::Int; seed::Int = 42)
    rng = Random.seed!(seed)
    patients = Matrix{Float64}(undef, n_samples, 10)
    loguniform(a, b) = exp(rand() * (log(b) - log(a)) + log(a))

    for i in 1:n_samples
        CV_mi      = rand(100:100:1000)
        CV_ao      = rand(200:200:2000)
        R_sys      = rand() * 2.0
        ratio_sys  = rand(5.0:0.5:20.0)
        R_svn      = clamp((R_sys + 0.55) / ratio_sys, 0.05, 0.5)
        C_sas      = loguniform(0.5, 2.5)
        ratio_comp = rand(5.0:0.5:25.0)
        C_svn      = ratio_comp * C_sas
        Elvmin     = loguniform(0.03, 0.15)
        ratio_lv   = rand(10.0:1.0:40.0)
        Elvmax     = ratio_lv * Elvmin
        Elamin     = loguniform(0.05, 0.2)
        ratio_la   = rand(3.0:1.0:15.0)
        Elamax     = ratio_la * Elamin
        if Elvmax <= Elamax; Elvmax = 1.2 * Elamax; end

        patients[i, :] = [CV_mi, CV_ao, R_sys, R_svn, C_svn, C_sas, Elvmax, Elvmin, Elamax, Elamin]
    end
    return patients
end

# =============================================================================
# 2. XML PARAMETER INJECTION
# =============================================================================
function set_xml_param!(doc, component_name, param_name, value)
    for node in eachelement(root(doc))
        if nodename(node) == "component" && node["name"] == component_name
            for var in eachelement(node)
                if nodename(var) == "variable" && var["name"] == param_name
                    var["initial_value"] = string(value)
                    return true
                end
            end
        end
    end
    error("Parameter not found: $component_name / $param_name")
end

function write_patient_params!(patient_params)
    # [CV_mi, CV_ao, R_scp, R_svn, C_svn, C_sas, Elvmax, Elvmin, Elamax, Elamin]

    heart_doc = readxml(HEART_FILE)
    set_xml_param!(heart_doc, "ParaHeart", "CVmi",   patient_params[1])
    set_xml_param!(heart_doc, "ParaHeart", "CVao",   patient_params[2])
    set_xml_param!(heart_doc, "ParaHeart", "ElvMax", patient_params[7])
    set_xml_param!(heart_doc, "ParaHeart", "ElvMin", patient_params[8])
    set_xml_param!(heart_doc, "ParaHeart", "ElaMax", patient_params[9])
    set_xml_param!(heart_doc, "ParaHeart", "ElaMin", patient_params[10])
    write(HEART_FILE, heart_doc)

    sys_doc = readxml(SYS_FILE)
    set_xml_param!(sys_doc, "ParaSys", "Rscp", patient_params[3])
    set_xml_param!(sys_doc, "ParaSys", "Rsvn", patient_params[4])
    set_xml_param!(sys_doc, "ParaSys", "Csvn", patient_params[5])
    set_xml_param!(sys_doc, "ParaSys", "Csas", patient_params[6])
    write(SYS_FILE, sys_doc)
end

# =============================================================================
# 3. SIMULATION
# =============================================================================
function simulate_patient(patient_params, tsteps)
    write_patient_params!(patient_params)

    # Suppress repetitive MTK warnings about ifelse valve logic at t=0
    ml   = with_logger(SimpleLogger(stderr, Logging.Error)) do
        CellModel(joinpath(BASE_PATH, "ModelMain.cellml"))
    end
    sys  = ml.sys
    prob = ODEProblem(ml, tspan)
    sol  = solve(prob, Tsit5(); saveat = tsteps, reltol = 1e-4, abstol = 1e-7)

    if sol.retcode != ReturnCode.Success
        @warn "Patient simulation did not converge!" retcode=sol.retcode params=patient_params
    end

    data = hcat(
        sol[sys.LV.Pi],   # pLV
        sol[sys.LA.Pi],   # pLA
        sol[sys.Sas.Pi],  # pSA
        sol[sys.Svn.Po],  # pSV
        sol[sys.LV.V],    # VLV
        sol[sys.LA.V],    # VLA
        sol[sys.LV.Qo],   # Qav
        sol[sys.LA.Qo],   # Qmv
        sol[sys.Sat.Qi],  # Qs
        sol[sys.Svn.Qi]   # Qsv
    )
    return data
end

# =============================================================================
# 4. SAVING
# Each patient gets its own file: data/patient_001.csv
# A single manifest file records all parameter values: data/patients_params.csv
# This scales cleanly to 1000+ patients — no single huge file.
# =============================================================================
function save_patient_data(patient_idx, data, params)
    # Time series data: rows = timesteps, cols = [pLV, pLA, pSA, pSV, VLV, VLA, Qav, Qmv, Qs, Qsv]
    filename = joinpath(DATA_PATH, @sprintf("patient_%04d.csv", patient_idx))
    open(filename, "w") do io
        # Header
        println(io, join(LABELS, ","))
        # Data
        writedlm(io, data, ',')
    end
end

function save_params_manifest(all_patients)
    filename = joinpath(DATA_PATH, "patients_params.csv")
    open(filename, "w") do io
        header = "patient_idx,CV_mi,CV_ao,R_scp,R_svn,C_svn,C_sas,Elvmax,Elvmin,Elamax,Elamin"
        println(io, header)
        for i in 1:size(all_patients, 1)
            row = vcat([i], all_patients[i, :])
            println(io, join(row, ","))
        end
    end
    println("Saved parameter manifest → $(filename)")
end

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
all_patients = generate_patients(NUMBER_OF_PATIENTS)

# Save parameter manifest upfront — safe even if simulation crashes midway
save_params_manifest(all_patients)

# Simulate and save each patient
all_patient_data = []
t_start = time()

for patient_idx in 1:NUMBER_OF_PATIENTS
    print("Simulating patient $patient_idx / $NUMBER_OF_PATIENTS ... ")
    t0 = time()

    data = simulate_patient(all_patients[patient_idx, :], tsteps)
    push!(all_patient_data, data)
    save_patient_data(patient_idx, data, all_patients[patient_idx, :])

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