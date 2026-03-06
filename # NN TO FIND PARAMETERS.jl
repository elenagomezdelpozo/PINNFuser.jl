# NN TO FIND PARAMETERS

using Flux
using DifferentialEquations
using DiffEqFlux
using LinearAlgebra
using DelimitedFiles
using Plots
# include("/Applications/Desktop/CODE/Thesis/PINNFuser.jl/examples/ElenasTry/cvmodelelena.jl")
using .elenasimpleModel
import Main.elenasimpleModel: Elastance_v, Elastance_a, DShiElastance_v, DShiElastance_a, Valve

τₑₛ_lv, τₑₚ_lv, τₑₛ_la, τₑₚ_la = 0.3, 0.45, 0.25, 0.15
Rmv0, Zao0, Rs0, Rsv0, Csa0, Csv0, Eₘₐₓ_lv0, Eₘᵢₙ_lv0, Eₘₐₓ_la0, Eₘᵢₙ_la0 = 0.012, 0.004, 1.01, 0.12, 1.3, 25.0, 2.7, 0.08, 0.25, 0.15
u0 = [7.0, 7.0, 7.0, 7.0, 200.0, 60.0, 0.0, 0.0, 0.0, 0.0] 
#.    pLV, pLA, psa, psv,  Vlv,  Vla,  Qav, Qmv, Qs, Qsv
τ = 1.0 # Cardiac cycle period

loaded_data = readdlm("/Applications/Desktop/CODE/Data_acquisition/data/original_data_new.txt")
original_data = Array{Float64}(loaded_data)  # select rows if needed
training_data = original_data[751:750 + 150, :]
x_input = reshape(training_data[:, :], :)  # vector of length num_steps*10
x_input = Float32.(x_input)   
input_size = size(x_input, 1)  # num_steps * 10

m = Flux.Chain(
    Flux.Dense(input_size, 32, tanh),
    Flux.Dense(32, 32, tanh),
    Flux.Dense(32, 10)
)

function predict(model, training_data)
    x_input = reshape(training_data[:, 1:10], :)       # flatten
    x_input = Float32.(x_input)                      # Flux requires Float32
    raw = model(x_input)

    Rmv = Rmv0 * (1 + 0.5 * tanh(raw[1]))
    Zao = Zao0 * (1 + 0.5 * tanh(raw[2]))
    Rs  = Rs0  * (1 + 0.5 * tanh(raw[3]))
    Rsv = Rsv0 * (1 + 0.5 * tanh(raw[4]))
    Csa = Csa0 * (1 + 0.5 * tanh(raw[5]))
    Csv = Csv0 * (1 + 0.5 * tanh(raw[6]))
    Eₘₐₓ_lv = Eₘₐₓ_lv0 * (1 + 0.5 * tanh(raw[7]))
    Eₘᵢₙ_lv = Eₘᵢₙ_lv0 * (1 + 0.5 * tanh(raw[8]))
    Eₘₐₓ_la = Eₘₐₓ_la0 * (1 + 0.5 * tanh(raw[9]))
    Eₘᵢₙ_la = Eₘᵢₙ_la0 * (1 + 0.5 * tanh(raw[10]))

    return [Rmv, Zao, Rs, Rsv, Csa, Csv, Eₘₐₓ_lv, Eₘᵢₙ_lv, Eₘₐₓ_la, Eₘᵢₙ_la]

end

function NIK_2ch!(du, u, p, t)
    pLV, pLA, psa, psv, Vlv, Vla, Qav, Qmv, Qs, Qsv = u
    Rmv, Zao, Rs, Rsv, Csa, Csv, Eₘₐₓ_lv, Eₘᵢₙ_lv, Eₘₐₓ_la, Eₘᵢₙ_la = p
    τₑₛ_lv, τₑₚ_lv, τₑₛ_la, τₑₚ_la = [0.3, 0.45, 0.9, 0.95]

    du[1] = (Qmv - Qav) * Elastance_v(t, Eₘᵢₙ_lv, Eₘₐₓ_lv, τ, τₑₛ_lv, τₑₚ_lv, 0.0) +
            (pLV / Elastance_v(t, Eₘᵢₙ_lv, Eₘₐₓ_lv, τ, τₑₛ_lv, τₑₚ_lv, 0.0)) *
            DShiElastance_v(t, Eₘᵢₙ_lv, Eₘₐₓ_lv, τ, τₑₛ_lv, τₑₚ_lv, 0.0)
    du[2] = (Qsv - Qmv) * Elastance_a(t, Eₘᵢₙ_la, Eₘₐₓ_la, τ, τₑₛ_la, τₑₚ_la) +
            (pLA / Elastance_a(t, Eₘᵢₙ_la, Eₘₐₓ_la, τ, τₑₛ_la, τₑₚ_la)) *
            DShiElastance_a(t, Eₘᵢₙ_la, Eₘₐₓ_la, τ, τₑₛ_la, τₑₚ_la)
    du[3] = (Qav - Qs) / Csa
    du[4] = (Qs - Qsv) / Csv
    du[5] = Qmv - Qav
    du[6] = Qsv - Qmv
    du[7] = Valve(Zao, du[1] - du[3], pLV - psa)
    du[8] = Valve(Rmv, du[2] - du[1], pLA - pLV)
    du[9] = (du[3] - du[4]) / Rs
    du[10] = (du[4] - du[2]) / Rsv
end

opt = Flux.ADAM(1e-4)
tspan = (5.0, 6.0)
num_steps = size(training_data,1)
saveat = range(tspan[1], tspan[2], length=num_steps)

data_vars = [1,2,3,4,5,6]

function loss_fn(model, training_data)
    p_pred = predict(model, training_data)
    prob = ODEProblem(NIK_2ch!, u0, tspan, p_pred)
    sol = solve(prob, Vern7(),
                saveat=saveat,
                reltol=1e-6,
                abstol=1e-6,
                sensealg=InterpolatingAdjoint())

    pred = Array(sol)'  
    pred_sel = pred[:, data_vars]
    data_sel = training_data[:, data_vars]
    return sum(abs2, pred_sel .- data_sel)
end


# ----------------------------
# Training loop
# ----------------------------
opt = Flux.Adam(1e-4)
opt_state = Flux.setup(opt, m)
prev_loss = 1e8
for epoch in 1:5000
    loss, grads = Flux.withgradient(m -> loss_fn(m, training_data), m)
    Flux.update!(opt_state, m, grads[1])
    if epoch % 50 == 0
        if abs(prev_loss-loss) < (1e-7 * loss) 
            println("Early stopping at $epoch, loss = $loss")
            break
        end
        #plot_prediction(m, epoch, training_data)
        println("Epoch $epoch, loss = $loss")
        global prev_loss = loss
    end
end

# After training
p_pred = predict(m, training_data)
println("Predicted params: ", p_pred)
