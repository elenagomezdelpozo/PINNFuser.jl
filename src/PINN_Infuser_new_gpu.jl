module PINNInfuser_new

__precompile__(false)   # fixes the rrule/ChainRules conflict

using Lux, LuxCUDA, CUDA
using StableRNGs, ComponentArrays, LinearAlgebra
using OrdinaryDiffEq, SciMLBase, SciMLSensitivity
using Statistics, Printf, Zygote
using Optimization, OptimizationOptimisers
using Plots

export PINN_Infuser_new

# ── Device setup (replaces deprecated Lux.gpu) ────────────────────────────
# gdev/cdev are callable objects: gdev(x) moves x to GPU, cdev(x) to CPU
const gdev = gpu_device()   # LuxDeviceUtils GPU device
const cdev = cpu_device()   # LuxDeviceUtils CPU device
_gpu_available() = CUDA.functional() 

function PINN_Infuser_new(
    ode_problem::SciMLBase.ODEProblem,
    ode_params,
    nn::Lux.Chain,
    training_steps::AbstractRange,
    target_data::AbstractMatrix{Float64};
    nn_output_weight::Float64  = 0.1,
    data_weight::Float64       = 1e-2,
    phy_weight::Float64        = 1e-4,
    optimizer                  = Adam,
    learning_rate::Float64     = 1e-3,
    iters::Int                 = 1000,
    early_stopping::Bool       = true,
    early_stop_window::Int     = 10,
    early_stop_warmup::Int     = 50,
    reltol::Float64            = 1e-6,
    abstol::Float64            = 1e-6,
    dtmax                      = Inf,
    rng::StableRNG             = StableRNG(5958),
    loss_logfile::String       = "training_logs/loss_history.txt",
    plot_every::Int            = 0,
    nn_vars::Union{Nothing,Vector{Int}}      = nothing,
    data_vars::Union{Nothing,Vector{Int}}    = nothing,
    physics_vars::Union{Nothing,Vector{Int}} = nothing,
)::Tuple{Any,Any,Any,Any}

    nn_vars      === nothing && error("`nn_vars` must be specified")
    data_vars    === nothing && error("`data_vars` must be specified")
    physics_vars === nothing && error("`physics_vars` must be specified")

    use_gpu = _gpu_available()
    @info "GPU available: $use_gpu  (CUDA=$(CUDA.functional()))"
    CUDA.functional()  && @info "CUDA device: $(CUDA.name(CUDA.device()))"

    U_MEAN    = vec(mean(target_data, dims=1))
    U_STD     = vec(std(target_data,  dims=1)) .+ 1e-6
    data_norm = (target_data .- U_MEAN') ./ U_STD'

    p_cpu, st_cpu = Lux.setup(rng, nn)
    p_cpu  = ComponentVector{Float32}(p_cpu) .* 1f-3

    p0     = p_cpu          # always CPU — solver and adjoint must stay on CPU
    st_dev = use_gpu ? gdev(st_cpu) : st_cpu   # state can still go to GPU

    function pinn_ode(u, p, t)
        du_base = Zygote.ignore() do
            du = zeros(Float64, length(u))
            ode_problem.f(du, collect(Float64.(u)), ode_params, t)
            du
        end

        u_in   = use_gpu ? gdev(Float32.(u)) : Float32.(u)
        p_dev      = use_gpu ? gdev(p) : p
        nn_out = nn(u_in, p_dev, st_dev)[1]
        nn_out_cpu = use_gpu ? cdev(nn_out) : nn_out

        n = length(u)
        contrib = sum(
            Float64.(Float32.(eachindex(u) .== nn_vars[k]) .* nn_out_cpu[k])
            for k in 1:length(nn_vars)
        )

        return du_base .+ contrib
    end

    # ── ODE predict ────────────────────────────────────────────────────────
    function predict(p)
        p_cpu = use_gpu ? cdev(p) : p
        prob = ODEProblem(
            pinn_ode,
            Float64.(ode_problem.u0),
            ode_problem.tspan,
            p,
        )
        sol =solve(prob, Vern7();
            saveat   = training_steps,
            dtmax    = dtmax,
            reltol   = reltol,
            abstol   = abstol,
            sensealg = InterpolatingAdjoint(autojacvec=ZygoteVJP()),
        )
        @info "✓ ODE solved."
        return sol
    end

    # ── Loss helpers ───────────────────────────────────────────────────────
    function _data_loss(pred_mat)
        pn = (pred_mat .- U_MEAN') ./ U_STD'
        sum(mean(abs2, pn[:,j] .- data_norm[:,j]) for j in data_vars)
    end

    function loss(p)
        @info "→ Computing loss..."
        sol      = predict(p)
        pred_mat = hcat(sol.u...)'
        total = data_weight * _data_loss(pred_mat)
        @info "✓ Loss computed: $total"
        return total
    end

    # ── Optimiser ──────────────────────────────────────────────────────────
    adtype  = Optimization.AutoZygote()
    optf    = Optimization.OptimizationFunction((x, _) -> loss(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, p0)

    losses     = Float64[]
    nn_history = Vector{Vector{Float64}}()
    var_names  = ["pLV","pLA","psa","psv","Vlv","Vla","Qav","Qmv","Qs","Qsv"]

    callback = function (state, _l)
        sol      = predict(state)           # was state.u
        pm       = Array(hcat(sol.u...)')
        L_data   = _data_loss(pm)
        total    = data_weight * L_data
        push!(losses, total)

        p_dev   = use_gpu ? gdev(state) : state   # was state.u
        contrib = zeros(length(nn_vars))
        for ti in 1:size(pm, 1)
            u_in = use_gpu ? gdev(Float32.(pm[ti,:])) : Float32.(pm[ti,:])
            out  = use_gpu ? cdev(nn(u_in, gdev(p_dev), st_dev)[1]) : nn(u_in, p_dev, st_dev)[1]
            for (k, _) in enumerate(nn_vars)
                contrib[k] += nn_output_weight * tanh(Float64(out[k]))
            end
        end
        push!(nn_history, contrib ./ size(pm, 1))

        @printf("Iter %4d | Total: %.5e | Data: %.5e\n", length(losses), total, L_data)

        if plot_every > 0 && length(losses) % plot_every == 0
            t    = sol.t
            plts = [plot(t, pm[:,i], label="PINN", lw=2, title=var_names[i]) for i in 1:min(10, size(pm,2))]
            for (j,i) in enumerate(data_vars)
                plot!(plts[i], t, target_data[:,j], label="Data", ls=:dash, lw=2)
            end
            display(plot(plts..., layout=(5,2), size=(900,800)))
        end

        if early_stopping && length(losses) > early_stop_warmup &&
           losses[end] > maximum(losses[max(1,end-early_stop_window):end-1])
            @info "Early stopping at iter $(length(losses))"
            return true
        end
        return false
    end

    trained = Optimization.solve(optprob, optimizer(learning_rate),
                                 callback=callback, maxiters=iters)

    folder = dirname(loss_logfile)
    !isempty(folder) && !isdir(folder) && mkpath(folder)
    open(loss_logfile, "w") do io
        for (i,L) in enumerate(losses); @printf(io,"%d %.12f\n",i,L); end
    end

    return (trained.u, st_cpu, losses, nn_history)
end

end # module LibInfuserNew