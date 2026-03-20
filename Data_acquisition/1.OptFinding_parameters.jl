using OrdinaryDiffEq
using OptimizationBBO
using LinearAlgebra
using DelimitedFiles
using Optimization, OptimizationOptimJL
 using Plots

if !isdefined(Main, :elenasimpleModel)
    include("/Applications/Desktop/CODE/PINNFuser.jl/examples/ElenasTry/cvmodelelena.jl")
    using .elenasimpleModel
    import Main.elenasimpleModel: Elastance_v, Elastance_a, DShiElastance_v, DShiElastance_a, Valve
end

τ = 1.0;  Eshift = 0.0

function NIK_2ch!(du, u, p, t)
    pLV, pLA, psa, psv, Vlv, Vla, Qav, Qmv, Qs, Qsv = u
    Rmv, Zao, Rs, Rsv, Csa, Csv, Eₘₐₓ_lv, Eₘᵢₙ_lv, Eₘₐₓ_la, Eₘᵢₙ_la = p[1:10]
    τₑₛ_lv, τₑₚ_lv, τₑₛ_la, τₑₚ_la = 0.3, 0.45, 0.90, 0.08

    E_lv  = Elastance_v(t,  Eₘᵢₙ_lv, Eₘₐₓ_lv, τ, τₑₛ_lv, τₑₚ_lv, Eshift)
    dE_lv = DShiElastance_v(t, Eₘᵢₙ_lv, Eₘₐₓ_lv, τ, τₑₛ_lv, τₑₚ_lv, Eshift)
    E_la  = Elastance_a(t,  Eₘᵢₙ_la, Eₘₐₓ_la, τ, τₑₛ_la, τₑₚ_la)
    dE_la = DShiElastance_a(t, Eₘᵢₙ_la, Eₘₐₓ_la, τ, τₑₛ_la, τₑₚ_la)

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

# ── Data —------──────────────────────────────────────────────────────────
raw_data      = readdlm("/Applications/Desktop/CODE/PINNFuser.jl/Data_acquisition/original_data_2Ch.txt")
training_data = Float64.(raw_data[1501:1500 + 30*150, 1:10])
println("Data loaded: $(size(training_data))")

# ── Problem setup ─────────────────────────────────────────────────────────────
num_cycles = 10
num_steps  = 150 * num_cycles   # 1500 points over the training window
warmup_end = 20.0
train_end  = 30.0               # warmup_end + num_cycles * 1.0
tspan      = (0.0, train_end)
data_vars  = 1:6
saveat     = range(warmup_end, train_end, length=num_steps)

# FIX 1: define fit_range — last 5 steady-state cycles
fit_cycles = 10
fit_range  = (num_steps - fit_cycles*150 + 1):num_steps

# FIX 2: compute max/min over the fit_range WINDOW, not a single row
target_max = vec(maximum(training_data[fit_range, data_vars], dims=1))
target_min = vec(minimum(training_data[fit_range, data_vars], dims=1))
target_amp = max.(target_max .- target_min, 1.0)   # FIX 3: floor at 1.0 to prevent /0
#         1Rmv    2Zao     3Rs     4Rsv    5Csa    6Csv    7Emax_lv 8Emin_lv 9Emax_la 10Emin_la 11pLV0 12pLA0  13psa0  14psv0  15Vlv0  16Vla0
p_phys = [0.009,  0.0020,  1.292,  0.070,  1.023,  10.90,  4.597,   0.0739,  0.14,    0.06,     8.0,   8.0,    30.0,   8.5,    130.0,  75.0]
lb     = [0.001,  0.0005,  0.500,  0.005,  0.500,  50.00,  0.500,   0.02,    0.05,    0.02,     2.0,   2.0,    60.0,   2.0,    80.0,   40.0]
ub     = [0.050,  0.0400,  2.500,  0.100,  2.500,  150.0,  5.000,   0.150,   0.40,    0.10,     15.0,  15.0,   120.0,  12.0,   180.0,  100.0]

slack    = 0.35
lb_tight = clamp.(p_phys .* (1 .- slack), lb, ub)
ub_tight = clamp.(p_phys .* (1 .+ slack), lb, ub)
maxmin_weights = [  1.0, 1.0,  1.0, 1.0,  1.0, 1.0]

function make_u0(p)
    return [p[11], p[12], p[13], p[14], p[15], p[16], 0.0, 0.0, 0.0, 0.0]
end

function run_model(p)
    u0    = make_u0(p)
    sol_w = solve(ODEProblem(NIK_2ch!, u0, (0.0, warmup_end), p),
                  Vern7(); reltol=1e-7, abstol=1e-7, save_everystep=false)
    (any(isnan, sol_w.u[end]) || any(x -> abs(x) > 1e5, sol_w.u[end])) && return nothing
 
    sol_t = solve(ODEProblem(NIK_2ch!, sol_w.u[end], tspan, p),
                  Vern7(); saveat=saveat, reltol=1e-7, abstol=1e-7)
    (any(isnan, sol_t.u[end]) ||
     any(x -> abs(x) > 1e5, sol_t.u[end]) ||
     length(sol_t.u) != num_steps) && return nothing
    return Array(sol_t)'   # (num_steps × 10)
end

function loss(p, _)
    pred = run_model(p)
    isnothing(pred) && return 1e10
 
    pred_window = pred[fit_range, data_vars]          # (fit_steps × 6)
    pred_max    = vec(maximum(pred_window, dims=1))   # (6,)
    pred_min    = vec(minimum(pred_window, dims=1))   # (6,)
 
    # Normalise errors by the target amplitude so all variables are on equal footing
    err_max = ((pred_max .- target_max) ./ target_amp) .* maxmin_weights
    err_min = ((pred_min .- target_min) ./ target_amp) .* maxmin_weights
 
    return sum(abs2, err_max) + sum(abs2, err_min)
end

# ── Sanity check ──────────────────────────────────────────────────────────────
l0 = loss(p_phys, nothing)
println("Loss at p_phys: $l0")
@assert isfinite(l0) && l0 < 1e9 "p_phys gives bad loss — check ODE/data!"


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — BBO global search (tight bounds)
# ══════════════════════════════════════════════════════════════════════════════
println("\n=== STAGE 1: BBO (max/min loss, tight bounds) ===")
 
optf_bbo    = Optimization.OptimizationFunction(loss)
optprob_bbo = Optimization.OptimizationProblem(optf_bbo, p_phys; lb=lb_tight, ub=ub_tight)
 
bbo_count  = Ref(0)
bbo_best   = Ref(l0)
bbo_best_p = copy(p_phys)
 
result_bbo = Optimization.solve(
    optprob_bbo,
    BBO_adaptive_de_rand_1_bin_radiuslimited();
    maxiters = 20_000,
    maxtime  = 180.0,
    callback = (state, lv) -> begin
        bbo_count[] += 1
        if lv < bbo_best[]
            bbo_best[]   = lv
            bbo_best_p  .= clamp.(state.u, lb_tight, ub_tight)
        end
        bbo_count[] % 2000 == 0 &&
            println("BBO iter $(bbo_count[]) | best=$(round(bbo_best[], sigdigits=5))")
        false
    end
)
 
p_bbo = bbo_best_p
println("BBO finished | best loss = $(bbo_best[])")
 
# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — LBFGS fine-tuning
# ══════════════════════════════════════════════════════════════════════════════
println("\n=== STAGE 2: LBFGS fine-tuning ===")
 
p_start = bbo_best[] < l0 ? p_bbo : p_phys
l_start = loss(p_start, nothing)
println("Starting loss: $l_start")
 
optf_loc    = Optimization.OptimizationFunction(loss, Optimization.AutoFiniteDiff())
optprob_loc = Optimization.OptimizationProblem(optf_loc, p_start; lb=lb_tight, ub=ub_tight)
 
loc_count = Ref(0)
best_loss = Ref(l_start)
best_p    = copy(p_start)
 
result_loc = Optimization.solve(
    optprob_loc,
    Fminbox(LBFGS());
    maxiters = 1000,
    callback = (state, lv) -> begin
        loc_count[] += 1
        p = clamp.(state.u, lb_tight, ub_tight)
        if lv < best_loss[]
            best_loss[] = lv
            best_p .= p
        end
        loc_count[] % 50 == 0 &&
            println("LBFGS iter $(loc_count[]) | loss=$(round(lv,sigdigits=5)) | best=$(round(best_loss[],sigdigits=5))")
        false
    end
)
 
p_opt = best_p
 
# ── Final comparison table ────────────────────────────────────────────────────
println("\n=== FINAL MAX/MIN COMPARISON ===")
pred_final  = run_model(p_opt)
pred_final_w = pred_final[fit_range, data_vars]
 
println("Variable |  2CH max  | 4CH max  | err%  ||  2CH min  | 4CH min  | err%")
println("-"^75)
for (i, name) in enumerate(varnames)
    pm  = maximum(pred_final_w[:, i])
    pn  = minimum(pred_final_w[:, i])
    tm  = target_max[i]
    tn  = target_min[i]
    em  = round((pm - tm) / tm * 100, digits=1)
    en  = round((pn - tn) / tn * 100, digits=1)
    println("  $(rpad(name,6)) | $(rpad(round(pm,digits=2),9)) | $(rpad(round(tm,digits=2),8)) | $(rpad(em,5))% || $(rpad(round(pn,digits=2),9)) | $(rpad(round(tn,digits=2),8)) | $(en)%")
end
 
println("\n=== LOSS SUMMARY ===")
println("Loss at p_phys   : $l0")
println("Loss after BBO   : $(bbo_best[])")
println("Loss after LBFGS : $(best_loss[])")
 
println("\n=== OPTIMISED PARAMETERS ===")
pnames = ["Rmv","Zao","Rs","Rsv","Csa","Csv","Emax_lv","Emin_lv","Emax_la","Emin_la",
          "τes_lv","τep_lv","τes_la","τep_la","pLV0","pLA0","psa0","psv0","Vlv0","Vla0"]
for (n, v, ref) in zip(pnames, p_opt, p_phys)
    Δ = round((v - ref) / ref * 100, digits=1)
    println("  $(rpad(n,10)) = $(rpad(round(v, sigdigits=5), 10))  (ref=$(round(ref,sigdigits=4)), Δ=$(Δ)%)")
end
