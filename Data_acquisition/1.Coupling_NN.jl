# NN TO FIND 2-CHAMBER PARAMETERS FROM 4-CHAMBER DATA
using Flux
using OrdinaryDiffEq
using DiffEqFlux
using LinearAlgebra
using DelimitedFiles, ForwardDiff
using Lux, Plots, Zygote, Statistics, StableRNGs, ComponentArrays
using Optim, Measures, BenchmarkTools
using SciMLSensitivity
using Plots, Printf
include("/Applications/Desktop/CODE/PINNFuser.jl/examples/ElenasTry/cvmodelelena.jl")
using .elenasimpleModel
import Main.elenasimpleModel: Elastance_v, Elastance_a, DShiElastance_v, DShiElastance_a, Valve

# ── Nominal (baseline) 2-chamber parameters ──────────────────────────────────
τₑₛ_lv, τₑₚ_lv, τₑₛ_la, τₑₚ_la = 0.3, 0.45, 0.25, 0.15
#Rmv0, Zao0, Rs0, Rsv0, Csa0, Csv0, Eₘₐₓ_lv0, Eₘᵢₙ_lv0, Eₘₐₓ_la0, Eₘᵢₙ_la0 = 0.012, 0.004, 1.01, 0.12, 1.3, 25.0, 2.7, 0.08, 0.25, 0.15
Rmv0, Zao0, Rs0, Rsv0, Csa0, Csv0, Eₘₐₓ_lv0, Eₘᵢₙ_lv0, Eₘₐₓ_la0, Eₘᵢₙ_la0  = [0.00911, 0.001, 1.3025, 0.085112, 1.423, 4.9930, 4.2973206, 0.0401941, 0.305619, 0.29]

# Initial conditions: pLV, pLA, psa, psv, Vlv, Vla, Qav, Qmv, Qs, Qsv
# u0 = [7.0, 7.0, 7.0, 7.0, 200.0, 60.0, 0.0, 0.0, 0.0, 0.0]
u0_0 = [7.5, 7.5, 7.5, 7.5, 210.0, 53.0, 0.0, 0.0, 0.0, 0.0]
τ = 1.0  # Cardiac cycle period

# ── 2-chamber ODE ─────────────────────────────────────────────────────────────
function NIK_2ch!(du, u, p, t)
    pLV, pLA, psa, psv, Vlv, Vla, Qav, Qmv, Qs, Qsv = u
    Rmv, Zao, Rs, Rsv, Csa, Csv, Eₘₐₓ_lv, Eₘᵢₙ_lv, Eₘₐₓ_la, Eₘᵢₙ_la = p
    τₑₛ_lv, τₑₚ_lv, τₑₛ_la, τₑₚ_la = 0.3, 0.45, 0.92, 0.09

    E_lv  = Elastance_v(t,  Eₘᵢₙ_lv, Eₘₐₓ_lv, τ, τₑₛ_lv, τₑₚ_lv, 0.0)
    dE_lv = DShiElastance_v(t, Eₘᵢₙ_lv, Eₘₐₓ_lv, τ, τₑₛ_lv, τₑₚ_lv, 0.0)
    E_la  = Elastance_a(t,  Eₘᵢₙ_la, Eₘₐₓ_la, τ, τₑₛ_la, τₑₚ_la)
    dE_la = DShiElastance_a(t, Eₘᵢₙ_la, Eₘₐₓ_la, τ, τₑₛ_la, τₑₚ_la)

    du[1] = (Qmv - Qav) * E_lv + (pLV / E_lv) * dE_lv
    du[2] = (Qsv - Qmv) * E_la + (pLA / E_la) * dE_la
    du[3] = (Qav - Qs)  / Csa
    du[4] = (Qs  - Qsv) / Csv
    du[5] = Qmv - Qav
    du[6] = Qsv - Qmv
    du[7] = Valve(Zao, du[1] - du[3], pLV - psa)
    du[8] = Valve(Rmv, du[2] - du[1], pLA - pLV)
    du[9] = (du[3] - du[4]) / Rs
    du[10] = (du[4] - du[2]) / Rsv
end

number_patients = 5

# ── Load 4-chamber data; take first 10 columns (matching 2-ch variables)
raw_data = readdlm("/Applications/Desktop/CODE/PINNFuser.jl/Data_acquisition/original_data_2Ch.txt")
training_data = Float32.(raw_data[751:900, 1:10])  # 150 rows × 10 cols

num_steps  = size(training_data, 1)   # 150
input_size = num_steps * 10           # flattened input to NN

# ── Network: maps flattened window → 10 parameter offsets ────────────
m = Flux.Chain(
    Flux.Dense(input_size, 32, tanh),
    Flux.Dense(32, 32,         tanh),
    Flux.Dense(32, 32,         tanh),
    Flux.Dense(32, 32,         tanh),
    Flux.Dense(32, 13)
)

# ── Decode NN output → physically bounded parameters ──────────────────
function predict(model, td)
    x = Float32.(reshape(td[:, 1:10], :))
    raw = model(x)

    Rmv     = Rmv0    * (1 + 0.5 * tanh(raw[1]))
    Zao     = Zao0    * (1 + 0.5 * tanh(raw[2]))
    Rs      = Rs0     * (1 + 0.5 * tanh(raw[3]))
    Rsv     = Rsv0    * (1 + 0.5 * tanh(raw[4]))
    Csa     = Csa0    * (1 + 0.5 * tanh(raw[5]))
    Csv     = Csv0    * (1 + 0.5 * tanh(raw[6]))
    Eₘₐₓ_lv = Eₘₐₓ_lv0 * (1 + 0.5 * tanh(raw[7]))
    Eₘᵢₙ_lv = Eₘᵢₙ_lv0 * (1 + 0.5 * tanh(raw[8]))
    Eₘₐₓ_la = Eₘₐₓ_la0 * (1 + 0.5 * tanh(raw[9]))
    Eₘᵢₙ_la = Eₘᵢₙ_la0 * (1 + 0.5 * tanh(raw[10]))

    p0      = u0_0[1] * (1 + 0.5 * tanh(raw[11]))  # bounded ~[3.75, 11.25]
    Vlv0    = u0_0[5] * (1 + 0.5 * tanh(raw[12]))  # bounded ~[105, 315]
    Vla0    = u0_0[6] * (1 + 0.5 * tanh(raw[13]))  # bounded ~[26, 80]

    u0 = [p0, p0, p0, p0, Vlv0, Vla0, 0.0, 0.0, 0.0, 0.0]


    return [Rmv, Zao, Rs, Rsv, Csa, Csv, Eₘₐₓ_lv, Eₘᵢₙ_lv, Eₘₐₓ_la, Eₘᵢₙ_la], u0
end

# ── Loss: solve 2-ch ODE with predicted params, compare to 4-ch data ──
tspan   = (5.0, 6.0)
saveat  = range(tspan[1], tspan[2], length=num_steps)
data_vars = 1:6
col_scale = Float32.(max.(sqrt.(mean(training_data[:, 1:6] .^ 2, dims=1)), 1f-3))

function loss_fn(model, training_data)
    p_pred, u0_pred = predict(model, training_data)
    prob = ODEProblem(NIK_2ch!, u0_pred, tspan, p_pred)
    sol = solve(prob, Vern7(),
                saveat=saveat,
                reltol=1e-6,
                abstol=1e-6,
                sensealg=InterpolatingAdjoint())

    pred = Array(sol)'  
    pred_sel = pred[:, data_vars]
    data_sel = training_data[:, data_vars]
    return sum(abs2, (pred_sel .- data_sel)./ col_scale)
end

opt = Flux.Adam(1e-4)
opt_state = Flux.setup(opt, m)
prev_loss = 1e8
for epoch in 1:5000
    loss, grads = Flux.withgradient(m -> loss_fn(m, training_data), m)
    Flux.update!(opt_state, m, grads[1])
    if epoch % 50 == 0
        if abs(prev_loss-loss) < (1e-4 * loss) 
            println("Early stopping at $epoch, loss = $loss")
            break
        end
        #plot_prediction(m, epoch, training_data)
        println("Epoch $epoch, loss = $loss")
        global prev_loss = loss
    end
end

# After training
p_pred, u0_pred = predict(m, training_data)
println("Predicted params: ", p_pred)
println("Predicted IC: ", u0_pred)


"""

# ── Per-patient training ───────────────────────────────────────────────────────
for i in 1:number_patients
    let
        println("\n===== Patient $i / $number_patients =====")
        # ── Load 4-chamber data; take first 10 columns (matching 2-ch variables)
        raw_data = readdlm(@sprintf("data/patient_%04d.csv", i), ',', Float64, skipstart=1)
        training_data = Float32.(raw_data[751:900, 1:10])  # 150 rows × 10 cols

        num_steps  = size(training_data, 1)   # 150
        input_size = num_steps * 10           # flattened input to NN

        # ── Network: maps flattened window → 10 parameter offsets ────────────
        m = Flux.Chain(
            Flux.Dense(input_size, 32, tanh),
            Flux.Dense(32, 32,     tanh),
            Flux.Dense(32, 10)
        )

        # ── Decode NN output → physically bounded parameters ──────────────────
        function predict(model, td)
            x = Float32.(reshape(td[:, 1:10], :))
            raw = model(x)

            Rmv    = Rmv0    * (1 + 0.5 * tanh(raw[1]))
            Zao    = Zao0    * (1 + 0.5 * tanh(raw[2]))
            Rs     = Rs0     * (1 + 0.5 * tanh(raw[3]))
            Rsv    = Rsv0    * (1 + 0.5 * tanh(raw[4]))
            Csa    = Csa0    * (1 + 0.5 * tanh(raw[5]))
            Csv    = Csv0    * (1 + 0.5 * tanh(raw[6]))
            Eₘₐₓ_lv = Eₘₐₓ_lv0 * (1 + 0.5 * tanh(raw[7]))
            Eₘᵢₙ_lv = Eₘᵢₙ_lv0 * (1 + 0.5 * tanh(raw[8]))
            Eₘₐₓ_la = Eₘₐₓ_la0 * (1 + 0.5 * tanh(raw[9]))
            Eₘᵢₙ_la = Eₘᵢₙ_la0 * (1 + 0.5 * tanh(raw[10]))

            return [Rmv, Zao, Rs, Rsv, Csa, Csv, Eₘₐₓ_lv, Eₘᵢₙ_lv, Eₘₐₓ_la, Eₘᵢₙ_la]
        end

        # ── Loss: solve 2-ch ODE with predicted params, compare to 4-ch data ──
        tspan   = (5.0, 6.0)
        saveat  = range(tspan[1], tspan[2], length=num_steps)
        data_vars = 1:6   # all 10 columns
        col_scale = Float32.(max.(sqrt.(mean(training_data[:, 1:6] .^ 2, dims=1)), 1f-3))

        function loss_fn(model, td)
            p_pred = predict(model, td)
            prob   = ODEProblem(NIK_2ch!, u0_default, tspan, p_pred)
            sol    = solve(prob, Tsit5();
                           saveat   = saveat,
                           reltol   = 1e-4,
                           abstol   = 1e-4,
                           sensealg = ForwardDiffSensitivity())

            sol.retcode != ReturnCode.Success && return Inf32

            pred = Float32.(Array(sol)')              # num_steps × 10
            residual = (pred[:, 1:6] .- td[:, 1:6]) ./ col_scale  # num_steps × 6
            return mean(abs2, residual)
        end

        # ── Training loop ─────────────────────────────────────────────────────
        opt       = Flux.Adam(1e-4)
        opt_state = Flux.setup(opt, m)
        prev_loss = 1f8

        for epoch in 1:10_000
            loss, grads = Flux.withgradient(m -> loss_fn(m, training_data), m)
            Flux.update!(opt_state, m, grads[1])

            if epoch % 1000 == 0
                println("  Epoch $epoch | loss = $loss")
                if abs(prev_loss - loss) < 1f-7 * abs(loss)
                    println("  Early stopping at epoch $epoch")
                    break
                end
                prev_loss = loss
            end
        end

        # ── Extract fitted parameters ─────────────────────────────────────────
        p_pred = predict(m, training_data)
        println("  Fitted params for patient $i: ", p_pred)

        # ── Extrapolate full 40-second simulation with fitted parameters ───────
        extrapolation_tspan      = (0.0, 40.0)
        num_of_samples_per_cycle = 150
        new_tsteps = range(extrapolation_tspan[1], extrapolation_tspan[2],
                           length = Int(floor(extrapolation_tspan[2] * num_of_samples_per_cycle)))

        ode_problem = ODEProblem(NIK_2ch!, u0_default, extrapolation_tspan, p_pred)
        ode_sol     = solve(ode_problem, Vern7();
                            saveat = new_tsteps,
                            reltol = 1e-6,
                            abstol = 1e-6)

        two_chamber_sol = Matrix(Array(ode_sol)')
        writedlm(@sprintf("data/two_chamber_%04d.txt", i), two_chamber_sol)
        println("  Saved → data/two_chamber_$(lpad(i,4,'0')).cvs")
    end
end

println("\nDone. All $number_patients patients processed.")
"""