using CellMLToolkit, ModelingToolkit, OrdinaryDiffEq
using DelimitedFiles, StableRNGs, Statistics
using Plots

rng = StableRNG(5958)

ml = CellModel("/Applications/Desktop/CODE/PINNFuser.jl/Data_acquisition/ModelMain.cellml")

# Training range
tspan = (0.0, 40.0)
num_of_samples = 150
tsteps = range(tspan[1], tspan[2], length = Int(tspan[2] * num_of_samples))

prob = ODEProblem(ml, tspan)
main_sol = solve(prob, Tsit5(); saveat = tsteps, reltol = 1e-4, abstol = 1e-7, dtmax = 1e-2)

sys = ml.sys

"""
# Data for 1 Chamber
data_to_save = hcat(
    main_sol[sys.LV.Pi],
    main_sol[sys.Sas.Pi],
    main_sol[sys.Svn.Po],
    main_sol[sys.LV.V],
    main_sol[sys.LV.Qo],
    main_sol[sys.LA.Qo],
    main_sol[sys.Svn.Qi],
)
writedlm("/Applications/Desktop/CODE/PINNFuser.jl/Data_acquisition/original_data_1Ch.txt", data_to_save)
println("Dane zapisane do pliku original_data_1Ch.txt")
"""
# Data for 2 Chambers
data_to_save = hcat(
    main_sol[sys.LV.Pi], # pLV
    main_sol[sys.LA.Pi], # pLA
    main_sol[sys.Sas.Pi], # pSA
    main_sol[sys.Svn.Po], # pSV
    main_sol[sys.LV.V], # vLV
    main_sol[sys.LA.V], # vLA
    main_sol[sys.LV.Qo], # Qav
    main_sol[sys.LA.Qo], # Qmv
    main_sol[sys.Sat.Qo], # Qs
    main_sol[sys.Svn.Qo], # Qsv
)
writedlm("/Applications/Desktop/CODE/PINNFuser.jl/Data_acquisition/original_data_2Ch.txt", data_to_save)
println("Dane zapisane do pliku original_data_2Ch.txt")

loaded_data = readdlm("/Applications/Desktop/CODE/PINNFuser.jl/Data_acquisition/original_data_2Ch.txt") # NEW DATA ACQUISITION METHOD
extrap_original_data = Array{Float64}(loaded_data)[1:Int(floor(40 * 150)), :] # 40 seconds of data, 150 samples per second
new_tseps = range(0, 20, length = 20*150)
mask_model = new_tseps .>= 18.0
time_to_plot = new_tseps[mask_model]
data_to_plot  = extrap_original_data[18*150+1:20*150 , :]


p1 = plot(
    time_to_plot,
    # [data_to_plot[:,1],
    data_to_plot[:,6],#],
    label = "VLA",
    lw = 2,
    ylims = (10,40)
)
display(p1)

p1 = plot(
    time_to_plot,
    # [data_to_plot[:,1],
    data_to_plot[:,2],#],
    label = "PLA",
    lw = 2,
    ylims = (1,5)
)
display(p1)

#P-V LOOP ATRIUM
p2 = plot(
    data_to_plot[:,6],
    data_to_plot[:,2],
    label = "PvsV",
    lw = 2
)
display(p2)
"""

# Add noise
# noise_magnitude = 0.00
# sd = std(data_to_save, dims = 2)
# noisy_data = data_to_save .+ (noise_magnitude*sd) .* randn(rng, eltype(data_to_save), size(data_to_save))

# writedlm("../data/original_extrapolation.txt", data_to_save)
# println("Dane zapisane do pliku original_extrapolation.txt")
"""