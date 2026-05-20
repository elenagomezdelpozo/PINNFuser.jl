module PatientsMod
# Creating_starting_ODE.jl
using Random
using OrdinaryDiffEq
using Lux, Zygote, Statistics, ComponentArrays
using Distributions
using StaticArrays

include("../src/Parameters.jl")
using .ParametersMod: parameters

include("../src/Model.jl")
using .ModelMod: NIK_2ch

export generate_patients

# time parameters are left constant (τₑₛ_lv, τₑₚ_lv, τₑₛ_la, τₑₚ_la) to avoid timing issues in the ODE solver
function generate_patients(n_samples::Int; seed::Union{Int,Nothing}=nothing) 
    seed !== nothing && Random.seed!(seed) # for reproducibility
    patients_params = Vector{Vector{Float64}}(undef, n_samples)
    loguniform(a, b) = exp(rand() * (log(b) - log(a)) + log(a))
    patients_odes = Vector{ODEProblem}()

    for i in 1:n_samples 
        R_mv  = loguniform(0.003, 0.025)
        R_ao  = loguniform(0.003, 0.020)
        R_s   = loguniform(0.5, 2.5)
        ratio_s = rand(Uniform(5.0, 15.0))
        R_sv  = clamp(R_s / ratio_s, 0.03, 0.4)
        C_sa  = loguniform(0.5, 2.5)
        ratio_comp = rand(Uniform(10.0, 30.0))
        C_sv  = ratio_comp * C_sa
        Elvmin = loguniform(0.03, 0.15)
        ratio_lv = rand(Uniform(10.0, 25.0))
        Elvmax   = ratio_lv * Elvmin
        Elvmax   = clamp(Elvmax, 1.0, 4.0)   # hard physiological cap
        Elamin = loguniform(0.06, 0.25)
        ratio_la = rand(Uniform(3.0, 8.0))
        Elamax   = ratio_la * Elamin
        Elamax   = clamp(Elamax, 0.1, 0.8)   # atrial max elastance cap
        if Elvmax <= Elamax
            Elvmax = min(1.5 * Elamax, 4.0)
        end

        params_vec = [R_mv, R_ao, R_s, R_sv, C_sa, C_sv, Elvmax, Elvmin, Elamax, Elamin]
        patients_params[i] = params_vec  
        ode_problem = ODEProblem(ModelMod.NIK_2ch, SA[parameters.u0...], parameters.tspan, params_vec);
        push!(patients_odes, ode_problem);
    end
    return patients_params, patients_odes
end # function

end # module
