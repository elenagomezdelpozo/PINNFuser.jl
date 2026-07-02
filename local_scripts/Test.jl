using DelimitedFiles

include("../src/Lib.jl")
using .LibInfuser
 
include("../src/Parameters.jl")
using .ParametersMod: parameters
 
name = parameters.name # name of the training to test
@info "Testing model: $(name)"

new_patient = parameters.original_all_data_list[3]   # aligned with parameters.plot_time (5 cycles)processed_all_data_list = map(process_patient_data, loaded_all_data_list, [Int(training.range_to_plot) for _ in 1:length(loaded_all_data_list)])processed_all_data_list = map(process_patient_data, loaded_all_data_list, [Int(training.range_to_plot) for _ in 1:length(loaded_all_data_list)])
LibInfuser.Tester_f(name, new_patient; test = true)
