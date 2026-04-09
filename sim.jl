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
                       ε=0.3, u_max=20.0, r_max=2.0, k_wall=50.0)
    N  = length(U_opt) + 1
    dt = tf_opt / (N - 1)

    # LQR gain at target equilibrium (used for stabilisation only)
    Q_lqr = diagm([1.0, 1.0, 100.0, 100.0, 10.0, 10.0])
    R_lqr = [0.1;;]
    A, B = linearise(p, x_eq)
    K    = lqr(A, B, Q_lqr, R_lqr)

    # Soft wall: spring-like force pushing cart back within [-r_max, r_max]
    # Applied on top of every control law to prevent runaway
    wall(r) = -k_wall * max(0.0, abs(r) - r_max) * sign(r)

    # Wrap raw state angles to [-π, π] before any control computation
    wrap(x) = [x[1], x[2], mod(x[3] + π, 2π) - π, mod(x[4] + π, 2π) - π, x[5], x[6]]

    lqr_ctrl = (x, _) -> clamp(-dot(K, wrap(x) - x_eq) + wall(x[1]), -u_max, u_max)

    return function (x, t)
        xw   = wrap(x)
        δ    = xw - x_eq

        if norm(δ) < ε
            return lqr_ctrl(x, t)
        end

        # Tracking: feedforward only + soft wall + saturation
        k    = clamp(floor(Int, t / dt) + 1, 1, length(U_opt))
        u_ff = U_opt[k][1]
        return clamp(u_ff + wall(x[1]), -u_max, u_max)
    end
end

# ---------------------------------------------------------------------------
# Simulate
# ---------------------------------------------------------------------------

tf_opt = 40.0   # must match tf used in swingup_opt.jl
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
δ_final = final_wrapped - x_eq

println("\n--- Simulation complete ---")
println("Steps:       ", length(sol.t))
println("Final state: ", round.(final_wrapped, digits=4))
println("Target:      ", round.(x_eq, digits=4))
println("Final error: ", round.(δ_final, digits=4))
println("Converged:   ", norm(δ_final) < 0.05)
