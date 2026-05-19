module TestMod

include("Model.jl")
using .ModelMod: NIK_2ch

include("Parameters.jl")
using .ParametersMod: parameters

export Tester_f, generate_patients

using DelimitedFiles
using JLD2          # ← add this — provides load()
using Lux
using StableRNGs
using OrdinaryDiffEq
using Plots
using ComponentArrays
using Random

function generate_patients(n_samples::Int; seed::Int)
    rng = Random.seed!(seed)
    patients = Matrix{Float64}(undef, n_samples, 10)
    loguniform(a, b) = exp(rand(rng) * (log(b) - log(a)) + log(a))

    for i in 1:n_samples
        Rmv  = loguniform(0.005, 0.05)
        Zao  = loguniform(0.001, 0.01)
        Rs   = loguniform(0.5, 2.5)
        Rsv  = loguniform(0.02, 0.15)
        Csa  = loguniform(0.5, 2.5)
        Csv  = loguniform(5.0, 20.0)
        Emin_lv = loguniform(0.03, 0.15)
        Emax_lv = Emin_lv * rand(rng, 10.0:1.0:40.0)
        if Emax_lv < 1.5; Emax_lv = 1.5; end  # hard floor for systolic function
        Emin_la = loguniform(0.03, 0.12)
        Emax_la = Emin_la * rand(rng, 3.0:0.5:10.0)
        if Emax_lv <= Emax_la; Emax_lv = 1.2 * Emax_la; end  # LV must dominate

        patients[i, :] = [Rmv, Zao, Rs, Rsv, Csa, Csv, Emax_lv, Emin_lv, Emax_la, Emin_la]
    end
    return patients
end

function Tester_f(name, new_ode_parameters)
    data = load(parameters.savepath)
    trained_st = data["trained_st"]
    trained_p = data["trained_p"]
    @info "Model loaded from $(parameters.savepath)"
    
    nn_vars = parameters.vars  
    @info "nn_vars: $(nn_vars)"

    # Reconstruct clean NamedTuple parameter structure (avoids ReshapedArray bug)
    rng = StableRNG(parameters.seed)
    trained_p, _ = Lux.setup(rng, parameters.NN)
    trained_p = ComponentVector{Float64}(trained_p)
    trained_p .= Float64.(data["trained_p"])  # copy values into fresh contiguous memory

    # ── PINN ODE (exactly mirrors pinn_ode! from training) ────────────────
    function pinn_ode(u, p, t)
        nn_output  = parameters.NN(u, p, st)[1]
        # physics doesn't depend on p_NN → ignore for gradient
        du_physics = Zygote.ignore() do
            Vector{Float64}(ode_problem.f(Vector{Float64}(u), ode_params, t))
        end
        correction = [i in nn_vars ?
                    parameters.nn_output_weight * nn_output[findfirst(==(i), nn_vars)] :
                    zero(eltype(nn_output))
                    for i in 1:length(u0_vec)]
        return du_physics .+ correction
    end
    
    # ── Solve ─────────────────────────────────────────────────────────────────
    pinn_problem = ODEProblem(
        (u, p, t) -> pinn_ode(u, trained_p, t),
        parameters.u0,
        parameters.tspan,
        trained_p,
    )
    solved_pinn = solve(pinn_problem, Vern7();
        saveat = parameters.plot_time, dtmax = parameters.dtmax, reltol = 1e-6, abstol = 1e-6)
    pinn_pred = Matrix(Array(solved_pinn)')   # (time × vars)

    # ── Baseline ODE (no NN) ──────────────────────────────────────────────────
    ode_prob_base = ODEProblem(ModelMod.NIK_2ch!, parameters.u0, parameters.tspan, new_ode_parameters)
    ode_sol       = solve(ode_prob_base, Vern7();
        saveat = parameters.plot_time, reltol = 1e-6, abstol = 1e-6)
    ode_pred = Matrix(Array(ode_sol)')

    # ── Plot ──────────────────────────────────────────────────────────────────
    @info "Plotting results for testing model \"$(name)\""
    plots = [
        begin
            p = plot(               
                pinn_pred[:, i],
                title     = parameters.labels[i],
                xlabel    = "time",
                ylabel    = parameters.units[i],
                ylims     = parameters.ylims[i],
                label = "PINN",
                lw = 2
            )
            plot!(
                p,
                ode_pred[:, i],
                label = "ODE",
                lw = 2,
                ls = :dash
            )
            plot!(
                p,
                parameters.extrap_original_data[Int(end-parameters.range_to_plot*parameters.num_of_samples_per_cycle+1):end, i],
                label = "DATA",
                lw = 2,
                ls = :dash
            )
            p
        end
        for i in 1:length(parameters.vars)
    ]
    p1 = plot(
        plots...,
        layout = (3, 2),
        size = (900, 800)
    )
    savefig(p1, "figures/testing_$(name).png")

end #function
end # module