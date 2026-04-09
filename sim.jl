# sim.jl — unified simulation: optimised swing-up → LQR stabilisation
#
# Workflow:
#   1. Run trajectory optimiser (swingup_opt.jl) to get open-loop U over [0, tf]
#   2. Interpolate U(t) for use inside the ODE
#   3. Switch to LQR once within ε of the target equilibrium
#   4. Simulate the full closed-loop trajectory via physics.jl

include("physics.jl")
include("swingup_opt.jl")   # defines X, U, x_target after solving

# ---------------------------------------------------------------------------
# Trajectory tracking controller
# ---------------------------------------------------------------------------
# Pure open-loop replaying U_opt diverges because the nonlinear ODE
# accumulates errors. Instead we use:
#
#   u(t) = U_opt(t) + wall(r)
#
# U_opt(t) is the feedforward from the optimiser (open-loop replay).
# wall(r) is a soft spring force keeping the cart within [-r_max, r_max].
# Using K feedback during swing-up was destabilising far from equilibrium,
# so tracking is feedforward-only with a wall to prevent runaway.
# Once within ε of the equilibrium, switch to pure LQR stabilisation.

function make_combined(p::Parameters, U_opt, tf_opt, x_eq;
                       ε=0.3, u_max=30.0, r_max=2.0, k_wall=50.0)
    N  = length(U_opt) + 1
    dt = tf_opt / (N - 1)

    # LQR gain at target equilibrium (used for stabilisation only)
    Q_lqr = diagm([1.0, 1.0, 100.0, 100.0, 10.0, 10.0])
    R_lqr = [0.1;;]
    A, B = linearise(p, x_eq)
    K    = try
        lqr(A, B, Q_lqr, R_lqr)
    catch err
        @warn "LQR solve failed; using feedforward tracking only" exception=(err, catch_backtrace())
        zeros(1, 6)
    end

    # Soft wall: spring-like force pushing cart back within [-r_max, r_max]
    # Applied on top of every control law to prevent runaway
    wall(r) = -k_wall * max(0.0, abs(r) - r_max) * sign(r)

    # Wrap raw state angles to [-π, π] before any control computation
    wrap(x) = [x[1], x[2], mod(x[3] + π, 2π) - π, mod(x[4] + π, 2π) - π, x[5], x[6]]

    # Compute state error with wrapped angular components.
    state_error(x, x_ref) = begin
        δ = copy(x) .- x_ref
        δ[3] = mod(δ[3] + π, 2π) - π
        δ[4] = mod(δ[4] + π, 2π) - π
        δ
    end

    x_eq_w = wrap(x_eq)

    # Linearly interpolate the discrete optimizer control samples to avoid
    # step changes in the force signal.
    interp_input(t) = begin
        τ = clamp(t / dt, 0.0, length(U_opt) - 1)
        k = clamp(floor(Int, τ) + 1, 1, length(U_opt))
        α = τ - (k - 1)
        if k == length(U_opt)
            return U_opt[end][1]
        end
        return (1 - α) * U_opt[k][1] + α * U_opt[k + 1][1]
    end

    lqr_ctrl = (x, _) -> clamp(-dot(K, state_error(wrap(x), x_eq_w)) + wall(x[1]), -u_max, u_max)
    
    # Check if LQR gain is valid (non-zero); fallback to multiaxis damping if solver failed
    has_lqr = maximum(abs.(K)) > sqrt(eps(Float64))
    
    # Fallback stabilizer: cart damping + angle feedback (since full LQR failed)
    fallback_ctrl = function (x, _)
        xw = wrap(x)
        δ = state_error(xw, x_eq_w)
        u_cart = -8.0*δ[1] - 6.0*δ[2]     # cart stabilization
        u_angle = -12.0*δ[3] - 10.0*δ[4] - 2.0*δ[5] - 2.0*δ[6]  # angle & rate damping
        clamp(u_cart + u_angle + wall(x[1]), -u_max, u_max)
    end
    
    stabilize_ctrl = has_lqr ? lqr_ctrl : fallback_ctrl

    return function (x, t)
        xw   = wrap(x)
        δ    = state_error(xw, x_eq_w)

        if t >= tf_opt || norm(δ) < ε
            return stabilize_ctrl(x, t)
        end

        # Tracking: feedforward only + soft wall + saturation
        u_ff = interp_input(t)
        return clamp(u_ff + wall(x[1]), -u_max, u_max)
    end
end

# ---------------------------------------------------------------------------
# Simulate
# ---------------------------------------------------------------------------

tf_opt = 36.0   # matches the selected successful trajectory from swingup_opt.jl
x_eq   = collect(Float64, x_target)   # convert SVector → Vector

p_phys = Parameters(m1_, m2_, M_, l1_, l2_, g_, no_control)
ctrl   = make_combined(p_phys, U, tf_opt, x_eq)
p_sim  = Parameters(p_phys.m1, p_phys.m2, p_phys.M,
                    p_phys.l1, p_phys.l2, p_phys.g, ctrl)

x_init = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # start from hanging down
prob   = ODEProblem(doublependulum!, x_init, (0.0, tf_opt + 10.0), p_sim)
sol    = solve(prob, Tsit5(), abstol=1e-8, reltol=1e-8)

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

final_wrapped = [sol.u[end][1], sol.u[end][2],
                 mod(sol.u[end][3] + π, 2π) - π,
                 mod(sol.u[end][4] + π, 2π) - π,
                 sol.u[end][5], sol.u[end][6]]
target_wrapped = [x_eq[1], x_eq[2],
                  mod(x_eq[3] + π, 2π) - π,
                  mod(x_eq[4] + π, 2π) - π,
                  x_eq[5], x_eq[6]]
δ_final = final_wrapped - target_wrapped
δ_final[3] = mod(δ_final[3] + π, 2π) - π
δ_final[4] = mod(δ_final[4] + π, 2π) - π

println("\n--- Simulation complete ---")
println("Steps:       ", length(sol.t))
println("Final state: ", round.(final_wrapped, digits=4))
println("Target:      ", round.(target_wrapped, digits=4))
println("Final error: ", round.(δ_final, digits=4))
println("Converged:   ", norm(δ_final) < 0.05)
