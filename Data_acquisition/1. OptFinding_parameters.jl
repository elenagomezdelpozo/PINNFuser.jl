using OrdinaryDiffEq
using OptimizationBBO
using LinearAlgebra
using DelimitedFiles
using Statistics
using Zygote
using SciMLSensitivity
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using Plots

if !isdefined(Main, :elenasimpleModel)
    include("/Applications/Desktop/CODE/PINNFuser.jl/examples/ElenasTry/cvmodelelena.jl")
    using .elenasimpleModel
    import Main.elenasimpleModel: Elastance_v, Elastance_a, DShiElastance_v, DShiElastance_a, Valve
end

# ── Fixed ─────────────────────────────────────────────────────────────────────
τ = 1.0
Eshift = 0.0

# ── ODE: timing comes from p[14:15] ──────────────────────────────────────────
function NIK_2ch!(du, u, p, t)
    pLV, pLA, psa, psv, Vlv, Vla, Qav, Qmv, Qs, Qsv = u
    Rmv, Zao, Rs, Rsv, Csa, Csv, Eₘₐₓ_lv, Eₘᵢₙ_lv, Eₘₐₓ_la, Eₘᵢₙ_la = p[1:10]
    τₑₛ_lv, τₑₚ_lv, τₑₛ_la, τₑₚ_la = 0.3, 0.45, 0.92, 0.09        # LV timing fixed

    E_lv   = Elastance_v(t,  Eₘᵢₙ_lv, Eₘₐₓ_lv, τ, τₑₛ_lv, τₑₚ_lv, Eshift)
    dE_lv  = DShiElastance_v(t, Eₘᵢₙ_lv, Eₘₐₓ_lv, τ, τₑₛ_lv, τₑₚ_lv, Eshift)
    E_la   = Elastance_a(t,  Eₘᵢₙ_la, Eₘₐₓ_la, τ, τₑₛ_la, τₑₚ_la)
    dE_la  = DShiElastance_a(t, Eₘᵢₙ_la, Eₘₐₓ_la, τ, τₑₛ_la, τₑₚ_la)

    du[1]  = (Qmv - Qav) * E_lv + (pLV / E_lv) * dE_lv
    du[2]  = (Qsv - Qmv) * E_la + (pLA / E_la) * dE_la
    du[3]  = (Qav - Qs)  / Csa
    du[4]  = (Qs  - Qsv) / Csv
    du[5]  = Qmv - Qav
    du[6]  = Qsv - Qmv
    du[7]  = Valve(Zao, du[1] - du[3], pLV - psa)
    du[8]  = Valve(Rmv, du[2] - du[1], pLA - pLV)
    du[9]  = (du[3] - du[4]) / Rs
    du[10] = (du[4] - du[2]) / Rsv
end

# ── Load and clean target data ────────────────────────────────────────────────
raw_data       = readdlm("/Applications/Desktop/CODE/PINNFuser.jl/Data_acquisition/original_data_2Ch.txt")
training_data  = Float64.(raw_data[1501:1500 + 6 * 150, 1:10])

# ── Problem setup ─────────────────────────────────────────────────────────────
# ── Problem setup ─────────────────────────────────────────────────────────────
num_steps  = 150 * 6                          
warmup_end = 5.0                          
train_end  = 11.0
tspan      = (0.0, 11.0)
data_vars  = 1:6
saveat     = range(warmup_end, train_end, length=num_steps)

col_scale = vec(maximum(training_data[:, data_vars], dims=1) .-
                minimum(training_data[:, data_vars], dims=1))
col_scale  = max.(col_scale, 1.0)

#          Rmv      Zao     Rs     Rsv     Csa    Csv  Emax_lv Emin_lv Emax_la Emin_la  p0   psa0   psv0   Vlv    Vla    
p_phys = [0.00911, 0.001, 1.3025, 0.0851, 1.423, 4.993, 4.297, 0.0402, 0.3056,  0.29,  7.5, 100.0,  10.0, 210.0, 63.0]
lb     = [0.001,   2e-4,  0.5,    0.01,   0.5,   1.0,   1.0,   0.01,   0.1,    0.05,   5.0,  60.0,   5.0, 100.0, 20.0]
ub     = [0.08,    0.01,  3.0,    0.4,    3.0,   15.0,  8.0,   0.2,    0.6,     0.4,  15.0, 140.0,  20.0, 350.0, 120.0]

weights    = [1.0, 5.0, 1.0, 2.0, 1.0, 3.0]
p_ref_norm = (p_phys .- lb) ./ (ub .- lb)
λ_reg      = 0.05   # ← add this; reduce to 0.01 if parameters aren't moving enough

function make_u0(p)
    # [pLV, pLA, psa,  psv,  Vlv,  Vla,  Qav, Qmv, Qs,  Qsv]
    return [p[11], p[11], p[12], p[13], p[14], p[15], 0.0, 0.0, 0.0, 0.0]
end

function run_model(p)
    p  = clamp.(p, lb, ub)
    u0 = make_u0(p)
    sol_w = solve(ODEProblem(NIK_2ch!, u0, (0.0, warmup_end), p),
                  Vern7(); reltol=1e-7, abstol=1e-7, save_everystep=false)
    any(isnan, sol_w.u[end]) && return nothing
    sol_t = solve(ODEProblem(NIK_2ch!, sol_w.u[end], tspan, p),
                  Vern7(); saveat=saveat, reltol=1e-7, abstol=1e-7)
    (any(isnan, sol_t.u[end]) || length(sol_t.u) != num_steps) && return nothing
    return Array(sol_t)'
end

function loss(p, _)
    pred = run_model(p)
    isnothing(pred) && return 1e10
    fit_range = (num_steps - 3*150 + 1):num_steps
    err       = (pred[fit_range, data_vars] .- training_data[fit_range, data_vars]) .*
                (weights[data_vars]' ./ col_scale')
    data_loss = sum(abs2, err) / (3*150)
    p_norm    = (clamp.(p, lb, ub) .- lb) ./ (ub .- lb)
    reg_loss  = sum(abs2, p_norm .- p_ref_norm) / length(p)
    return data_loss + λ_reg * reg_loss
end

# ── Verify starting point is sane ─────────────────────────────────────────────
l0 = loss(p_phys, nothing)
println("Loss at p_phys: $l0")
@assert isfinite(l0) && l0 < 1e9 "p_phys gives bad loss — check ODE!"

# ── Optimise ──────────────────────────────────────────────────────────────────
optf    = Optimization.OptimizationFunction(loss, Optimization.AutoFiniteDiff())
optprob = Optimization.OptimizationProblem(optf, p_phys; lb=lb, ub=ub)

iter_count = Ref(0)
best_loss  = Ref(l0)        # ← start at p_phys loss, not Inf
best_p     = copy(p_phys)

result = Optimization.solve(
    optprob,
    Fminbox(LBFGS());
    maxiters = 500,
    callback = (state, lv) -> begin
        iter_count[] += 1
        p = clamp.(state.u, lb, ub)
        if lv < best_loss[]
            best_loss[] = lv
            best_p .= p
        end
        iter_count[] % 1 == 0 &&
            println("Iter $(iter_count[]) | loss=$(round(lv,sigdigits=5)) | best=$(round(best_loss[],sigdigits=5))")
        false
    end
)

p_opt = best_p
println("\nFinal best loss : ", best_loss[])
println("Loss at p_phys  : ", l0)
println("p_opt = ", p_opt)