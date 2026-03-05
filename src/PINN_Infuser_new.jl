module PINNInfuser_new

__precompile__(false)   # fixes the rrule/ChainRules conflict

using Lux, LuxCUDA, CUDA
using StableRNGs, ComponentArrays, LinearAlgebra
using Optimization, OptimizationOptimisers
using SciMLBase, SciMLSensitivity
using Statistics, ForwardDiff
using OrdinaryDiffEq: Vern7, Tsit5, Rodas4
using Printf
using Plots, Zygote, Statistics

export PINN_Infuser_new

# ── Device setup (replaces deprecated Lux.gpu) ────────────────────────────
# gdev/cdev are callable objects: gdev(x) moves x to GPU, cdev(x) to CPU
const gdev = gpu_device()   # LuxDeviceUtils GPU device
const cdev = cpu_device()   # LuxDeviceUtils CPU device
_gpu_available() = CUDA.functional() 

"""

Trains a Physics-Informed Neural Network (PINN) by minimizing a composite loss function
that includes both data fidelity and physical law adherence.

# Arguments
- `ode_problem::SciMLBase.ODEProblem`: The ODE problem defining the physical laws.
- `nn::Lux.Chain`: The Lux neural network model to be trained.
- `training_steps::AbstractRange`: The time steps at which to evaluate the solution and calculate loss function.
- `target_data::Array{Float64}`: The ground truth data for training.

# Keyword Arguments
- `early_stopping::Bool = true`: Whether to enable early stopping based on loss convergence.
- `nn_output_weight::Float64 = 0.1`: The weight factor for the NN infusion in ODE.
- `physics_weight::Float64 = 1.0`: The weight of the physics-based loss component.
- `optimizer = OptimizationOptimisers.Adam`: The optimization algorithm to use.
- `learning_rate::Float64 = 0.001`: The learning rate for the optimizer.
- `reltol::Float64 = 1e-6`: The relative tolerance for the ODE solver.
- `abstol::Float64 = 1e-6`: The absolute tolerance for the ODE solver.
- `dtmax = Inf`: The maximum time step for the ODE solver.
- `iters::Int = 1000`: The number of training iterations.
- `rng::StableRNG` = StableRNG(5958): A random number generator for reproducibility.
- `loss_logfile::String = "training_logs/loss_history.txt"`: File path to log loss history.
- `data_vars::Union{Nothing,Vector{Int}} = nothing`: Indices of variables to include in data loss.
- `physics_vars::Union{Nothing,Vector{Int}} = nothing`: Indices of variables to include in physics loss.

# Returns
- `Tuple{Any, Any}`: The trained parameters of the neural network.
"""

function PINN_Infuser_new(
    ode_problem::SciMLBase.ODEProblem,
    ode_params,
    nn::Lux.Chain,
    training_steps::AbstractRange,
    target_data::AbstractMatrix{Float64};
    processor,
    early_stopping::Bool = true,
    nn_output_weight::Float64 = 0.1,
    physics_weight::Float64 = 1.0,
    optimizer = ADAM,
    learning_rate::Float64 = 1e-3,
    reltol::Float64 = 1e-6,
    abstol::Float64 = 1e-6,
    dtmax = Inf,
    iters::Int = 1000,
    plot_every::Int = 10,
    early_stop_warmup::Int = 50,
    early_stop_window::Int = 10,
    rng::StableRNG = StableRNG(5958),
    loss_logfile::String = "training_logs/loss_history.txt",
    nn_vars::Union{Nothing,Vector{Int}} = nothing,
    data_vars::Union{Nothing,Vector{Int}} = nothing,
    physics_vars::Union{Nothing,Vector{Int}} = nothing,
)::Tuple{Any,Any, Any, Any}
    if processor == "cpu"
        U_MEAN = vec(mean(target_data, dims = 1))
        U_STD = vec(std(target_data, dims = 1)) .+ 1e-6
        data_norm = (target_data .- U_MEAN') ./ U_STD'

        p_NN, st = Lux.setup(rng, nn)
        p_NN = 1e-3 * ComponentVector{Float64}(p_NN)

        function cpu_pinn_ode!(du, u, p_NN, t)
            nn_output = nn(u, p_NN, st)[1]
            ode_problem.f(du, u, ode_params, t)
            for (k, i) in enumerate(nn_vars)
                du[i] *= 1 .+ nn_output_weight .* tanh.(nn_output[k])
            end    
        end

        function cpu_predict(p_NN)
            prob = ODEProblem(
                (du, u, p, t) -> cpu_pinn_ode!(du, u, p, t), 
                ode_problem.u0,
                ode_problem.tspan,
                p_NN,
            )
            temp_sol = solve(
                prob, 
                Vern7(),
                saveat=training_steps,
                dtmax=dtmax,
                reltol=reltol,
                abstol=abstol
            )
            return temp_sol
        end

        function cpu_compute_nn_contributions(sol, p_NN)
            t = sol.t
            u_mat = hcat(sol.u...)'  # (time × variables)
            nn_contrib = zeros(length(nn_vars))
            for (k, i) in enumerate(nn_vars)
                nn_outputs_over_time = [
                    nn(u_mat[ti, :], p_NN, st)[1][k] for ti in 1:length(t)
                ]
                nn_contrib[i] = nn_output_weight * sum(nn_outputs_over_time)
            end
            return nn_contrib
        end

        function cpu_data_loss(pred_norm, data_norm, data_vars)
            return sum(mean(abs2, pred_norm[:, j] .- data_norm[:, j]) for j in data_vars)
        end

        function cpu_physics_loss(pred, p_NN, physics_vars)
            pred_mat = hcat(pred.u...)'
            l = 0.0
            for (i, t) in enumerate(training_steps)
                u = pred_mat[i, :]
                du = similar(u)
                cpu_pinn_ode!(du, u, p_NN, t)
                f_base = similar(u)
                ode_problem.f(f_base, u, ode_params, t)
                l += mean(abs2.(du[physics_vars] .- f_base[physics_vars]))
            end
            return l / length(training_steps)
        end

        function cpu_loss(p_NN)
            pred = cpu_predict(p_NN)
            pred_mat = hcat(pred.u...)'
            pred_norm = (pred_mat .- U_MEAN') ./ U_STD'
            L_data = cpu_data_loss(pred_norm, data_norm, data_vars)
            L_phy = cpu_physics_loss(pred, p_NN, physics_vars)        
            return 1e-2 * L_data + 1e-4 * L_phy
        end

        adtype = Optimization.AutoForwardDiff()
        optf = Optimization.OptimizationFunction((x, p) -> cpu_loss(x), adtype)
        optprob = Optimization.OptimizationProblem(optf, p_NN)
        losses = Float64[]
        nn_history = Vector{Vector{Float64}}()

        callback = function (state, l)
            pred = cpu_predict(state)
            pred_mat = hcat(pred.u...)'
            pred_norm = (pred_mat .- U_MEAN') ./ U_STD'
            l_data = cpu_data_loss(pred_norm, data_norm, data_vars)
            l_phy = cpu_physics_loss(pred, state, physics_vars)
            nn_contrib = cpu_compute_nn_contributions(pred, state)
            push!(losses, 1e-2 * l_data + l_phy)
                push!(nn_history, nn_contrib) 
                println(
                "Iter $(length(losses)) | " *
                "Total: $(round(1e-2 * l_data + 1e-4 * l_phy, sigdigits=5)) | " *
                "Data: $(round(l_data, sigdigits=5)) | " *
                "Physics: $(round(l_phy, sigdigits=5)) | " 
            )
            if early_stopping &&
                length(losses) > 50 &&
                losses[end] - maximum(losses[(end-10):(end-1)]) > 0
                println("Early stopping at iteration $(length(losses)) with loss $(losses[end])")
                return true
            else
                return false
            end
        end

        trained_params = Optimization.solve(
            optprob,
            optimizer(learning_rate),
            callback = callback,
            maxiters = iters,
        )

        folder = dirname(loss_logfile)
        if folder != "" && !isdir(folder)
            println("Creating directory for training logs: $folder")
            mkpath(folder)
        end
        open(loss_logfile, "w") do io
            for (i, L) in enumerate(losses)
                @printf(io, "%d %.12f\n", i, L)
            end
        end

        return (trained_params.u, st, losses, nn_history)    

    elseif processor == "gpu"
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

        function gpu_pinn_ode(u, p, t)
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
        function gpu_predict(p)
            p = use_gpu ? cdev(p) : p
            prob = ODEProblem(
                gpu_pinn_ode,
                Float64.(ode_problem.u0),
                ode_problem.tspan,
                p,
            )
            sol = solve(
                prob, 
                Vern7(),
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
        function gpu_data_loss(pred_mat)
            pn = (pred_mat .- U_MEAN') ./ U_STD'
            sum(mean(abs2, pn[:,j] .- data_norm[:,j]) for j in data_vars)
        end

        function gpu_loss(p)
            @info "→ Computing loss..."
            sol      = gpu_predict(p)
            pred_mat = hcat(sol.u...)'
            total = data_weight * gpu_data_loss(pred_mat)
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
            sol      = gpu_predict(state)           # was state.u
            pm       = Array(hcat(sol.u...)')
            L_data   = gpu_data_loss(pm)
            total    = data_weight * L_data
            push!(losses, total)

            p_dev   = use_gpu ? gdev(state) : state   # was state.u
            contrib = zeros(length(nn_vars))
            for ti in 1:size(pm, 1)
                u_in = use_gpu ? gdev(Float32.(pm[ti,:])) : Float32.(pm[ti,:])
                out  = use_gpu ? cdev(nn(u_in, gdev(p_dev), st_dev)[1]) : nn(u_in, p_dev, st_dev)[1]
                for (k, _) in enumerate(nn_vars)
                    contrib[k] += nn_output_weight * Float64(out[k])
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
    else
        error("Invalid processor specified: $processor. Use 'cpu' or 'gpu'.")
    end # if 

end # function

end # module PINNInfuser