using CellMLToolkit, ModelingToolkit, OrdinaryDiffEq
using DelimitedFiles, StableRNGs, Statistics

rng = StableRNG(5958)
# https://models.cellml.org/workspace/shi_hose_2009
ml = CellModel("4Chamber/ModelMain.cellml")

# Training range
tspan = (0.0, 20.0)
num_of_samples = 300
tsteps = range(5.0, 7.0, length = num_of_samples)
# num_of_samples = 3000
# tsteps = range(0.0, 20.0, length = num_of_samples)

prob = ODEProblem(ml, tspan)
main_sol = solve(prob, Tsit5(); saveat = tsteps, reltol = 1e-4, abstol = 1e-7, dtmax = 1e-2)

sys = ml.sys

data_to_save = hcat(
    main_sol[sys.LV.Pi],
    main_sol[sys.Sas.Pi],
    main_sol[sys.Svn.Po],
    main_sol[sys.LV.V],
    main_sol[sys.LV.Qo],
    main_sol[sys.LA.Qo],
    main_sol[sys.Svn.Qi],
)
labels = [
    "LA_P", "LA_V", "LA_Qo",
    "LV_P", "LV_V", "LV_Qo",
    "RA_P", "RA_V", "RA_Qo",
    "RV_P", "RV_V", "RV_Qo",
]

# Add noise
# noise_magnitude = 0.00
# sd = std(data_to_save, dims = 2)
# noisy_data = data_to_save .+ (noise_magnitude*sd) .* randn(rng, eltype(data_to_save), size(data_to_save))

writedlm("../data/original_data_new.txt", data_to_save)
println("Dane zapisane do pliku original_data.txt")

# writedlm("../data/original_extrapolation.txt", data_to_save)
# println("Dane zapisane do pliku original_extrapolation.txt")