__precompile__(false)  # Add this here
module ModelMod

include("Model_supporting_f.jl")
using .ModelSuportMod
using .ModelSuportMod: Valve, Elastance_v, Elastance_a, DShiElastance_v, DShiElastance_a
using StaticArrays

export NIK_2ch
export NIK_2ch!

function NIK_2ch(u::AbstractVector, p, t) # out of place (no !) for Zygote AD, in place (!) for ODE solvers
    pLV, pLA, psa, psv, Vlv, Vla, Qav, Qmv, Qs, Qsv = u
    Rmv, Zao, Rs, Rsv, Csa, Csv, Eₘₐₓ_lv, Eₘᵢₙ_lv, Eₘₐₓ_la, Eₘᵢₙ_la = p
    τ, τₑₛ_lv, τₑₚ_lv, τₑₛ_la, τₑₚ_la = 1.0, 0.3, 0.45, 0.92, 0.09
    Eshift = 0.0

    E_lv  = Elastance_v(t,  Eₘᵢₙ_lv, Eₘₐₓ_lv, τ, τₑₛ_lv, τₑₚ_lv, Eshift)
    dE_lv = DShiElastance_v(t, Eₘᵢₙ_lv, Eₘₐₓ_lv, τ, τₑₛ_lv, τₑₚ_lv, Eshift)
    E_la  = Elastance_a(t,  Eₘᵢₙ_la, Eₘₐₓ_la, τ, τₑₛ_la, τₑₚ_la)
    dE_la = DShiElastance_a(t, Eₘᵢₙ_la, Eₘₐₓ_la, τ, τₑₛ_la, τₑₚ_la)

    du1 = (Qmv - Qav) * E_lv + (pLV / E_lv) * dE_lv # Left Ventricle pressure
    du2 = (Qsv - Qmv) * E_la + (pLA / E_la) * dE_la # Left Atrium pressure
    du3 = (Qav - Qs) / Csa # Systemic Arterial pressure
    du4 = (Qs - Qsv) / Csv # Systemic Venous pressure
    du5 = Qmv - Qav # LV volume
    du6 = Qsv - Qmv # LA volume
    du7 = Valve(Zao, du1 - du3, pLV- psa)  # AV flow
    du8 = Valve(Rmv, du2 - du1, pLA - pLV)  # MV flow
    du9 = (du3 - du4) / Rs # Systemic flow
    du10 = (du4 - du2) / Rsv # Venous flow

    return SVector{10}(du1, du2, du3, du4, du5, du6, du7, du8, du9, du10)
end

function NIK_2ch!(du, u::AbstractVector, p, t) # out of place (no !) for Zygote AD, in place (!) for ODE solvers
    pLV, pLA, psa, psv, Vlv, Vla, Qav, Qmv, Qs, Qsv = u
    Rmv, Zao, Rs, Rsv, Csa, Csv, Eₘₐₓ_lv, Eₘᵢₙ_lv, Eₘₐₓ_la, Eₘᵢₙ_la = p
    τ, τₑₛ_lv, τₑₚ_lv, τₑₛ_la, τₑₚ_la = 1.0, 0.3, 0.45, 0.92, 0.09
    Eshift = 0.0

    E_lv  = Elastance_v(t,  Eₘᵢₙ_lv, Eₘₐₓ_lv, τ, τₑₛ_lv, τₑₚ_lv, Eshift)
    dE_lv = DShiElastance_v(t, Eₘᵢₙ_lv, Eₘₐₓ_lv, τ, τₑₛ_lv, τₑₚ_lv, Eshift)
    E_la  = Elastance_a(t,  Eₘᵢₙ_la, Eₘₐₓ_la, τ, τₑₛ_la, τₑₚ_la)
    dE_la = DShiElastance_a(t, Eₘᵢₙ_la, Eₘₐₓ_la, τ, τₑₛ_la, τₑₚ_la)

    du[1] = (Qmv - Qav) * E_lv + (pLV / E_lv) * dE_lv # Left Ventricle pressure
    du[2] = (Qsv - Qmv) * E_la + (pLA / E_la) * dE_la # Left Atrium pressure
    du[3] = (Qav - Qs) / Csa # Systemic Arterial pressure
    du[4] = (Qs - Qsv) / Csv # Systemic Venous pressure
    du[5] = Qmv - Qav # LV volume
    du[6] = Qsv - Qmv # LA volume
    du[7] = Valve(Zao, du[1] - du[3], pLV- psa)  # AV flow
    du[8] = Valve(Rmv, du[2] - du[1], pLA - pLV)  # MV flow
    du[9] = (du[3] - du[4]) / Rs # Systemic flow
    du[10] = (du[4] - du[2]) / Rsv # Venous flow
end

end