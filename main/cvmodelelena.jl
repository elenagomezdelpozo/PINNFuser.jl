module elenasimpleModel

export NIK_2ch, Valve, Elastance_v, Elastance_a, DShiElastance_v, DShiElastance_a

function Valve(R, deltaP, open)
    dq = 0.0
    if (open) > 0.0
        dq = deltaP / R
    else
        dq = 0.0
    end
    return dq

end
function Elastance_v(t, Eₘᵢₙ, Eₘₐₓ, τ, τₑₛ, τₑₚ, Eshift)
    τₑₛ = τₑₛ * τ

    τₑₚ = τₑₚ * τ
    #τ = 4/3(τₑₛ+τₑₚ)
    tᵢ = rem(t + (1 - Eshift) * τ, τ)

    Eₚ =
        (tᵢ <= τₑₛ) * (1 - cos(tᵢ / τₑₛ * pi)) / 2 +
        (tᵢ > τₑₛ) * (tᵢ <= τₑₚ) * (1 + cos((tᵢ - τₑₛ) / (τₑₚ - τₑₛ) * pi)) / 2 +
        (tᵢ <= τₑₚ) * 0

    E = Eₘᵢₙ + (Eₘₐₓ - Eₘᵢₙ) * Eₚ

    return E 
end
function DShiElastance_v(t, Eₘᵢₙ, Eₘₐₓ, τ, τₑₛ, τₑₚ, Eshift)

    τₑₛ = τₑₛ * τ
    τₑₚ = τₑₚ * τ
    #τ = 4/3(τₑₛ+τₑₚ)
    tᵢ = rem(t + (1 - Eshift) * τ, τ)

    DEₚ =
        (tᵢ <= τₑₛ) * pi / τₑₛ * sin(tᵢ / τₑₛ * pi) / 2 +
        (tᵢ > τₑₛ) * (tᵢ <= τₑₚ) * pi / (τₑₚ - τₑₛ) * sin((τₑₛ - tᵢ) / (τₑₚ - τₑₛ) * pi) / 2
    (tᵢ <= τₑₚ) * 0
    DE = (Eₘₐₓ - Eₘᵢₙ) * DEₚ

    return DE 
end

function Elastance_a(t, Eₘᵢₙ, Eₘₐₓ, τ, τₑₛ, τₑₚ)
    tₙ = rem(t, τ)
    if τₑₛ < tₙ <= τₑₚ
        eₚ = 1 - cos(2 * pi * (tₙ - τₑₛ) / (τₑₛ - τₑₚ))
    else
        eₚ = 0.0
    end
    return Eₘᵢₙ + 0.5 * (Eₘₐₓ - Eₘᵢₙ) * eₚ
end

function DShiElastance_a(t, Eₘᵢₙ, Eₘₐₓ, τ, τₑₛ, τₑₚ)
    tₙ = rem(t, τ)
    if τₑₛ < tₙ <= τₑₚ
        deₚ = sin((2 * pi / (τₑₛ - τₑₚ)) * (tₙ - τₑₛ)) * (2 * pi / (τₑₛ - τₑₚ))
        return 0.5 * (Eₘₐₓ - Eₘᵢₙ) * deₚ
    else
        return 0.0
    end
end

function NIK_2ch!(du, u, p, t)
    pLV, pLA, psa, psv, Vlv, Vla, Qav, Qmv, Qs, Qsv = u
    Rmv, Zao, Rs, Rsv, Csa, Csv, Eₘₐₓ_lv, Eₘᵢₙ_lv, Eₘₐₓ_la, Eₘᵢₙ_la = p

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