# visualise.jl — plots and animation for the double pendulum simulation
#
# Run after sim.jl has been included (sol, x_eq, p_phys must be in scope).
# Install:  ]add Plots

using Plots

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

wrap_angle(θ) = mod.(θ .+ π, 2π) .- π

function extract(sol)
    t    = sol.t
    r    = [u[1] for u in sol.u]
    dr   = [u[2] for u in sol.u]
    θ1   = wrap_angle([u[3] for u in sol.u])
    θ2   = wrap_angle([u[4] for u in sol.u])
    dθ1  = [u[5] for u in sol.u]
    dθ2  = [u[6] for u in sol.u]
    return t, r, dr, θ1, θ2, dθ1, dθ2
end

function compute_control(sol, p, x_eq; u_max=20.0, r_max=2.0, k_wall=50.0)
    # Recompute u at each timestep for plotting
    # (same logic as make_combined in sim.jl)
    Q_lqr = diagm([1.0, 1.0, 100.0, 100.0, 10.0, 10.0])
    R_lqr = [0.1;;]
    A, B  = linearise(p, x_eq)
    K     = lqr(A, B, Q_lqr, R_lqr)
    wall  = r -> -k_wall * max(0.0, abs(r) - r_max) * sign(r)
    wrap  = x -> [x[1], x[2], mod(x[3]+π,2π)-π, mod(x[4]+π,2π)-π, x[5], x[6]]

    return [clamp(-dot(K, wrap(u) - x_eq) + wall(u[1]), -u_max, u_max)
            for u in sol.u]
end

# ---------------------------------------------------------------------------
# State plots
# ---------------------------------------------------------------------------

function plot_states(sol, x_eq)
    t, r, dr, θ1, θ2, dθ1, dθ2 = extract(sol)

    p1 = plot(t, r,   label="r (cart pos)",  ylabel="m",   color=:black)
    hline!(p1, [0.0], linestyle=:dash, color=:gray, label="target")

    p2 = plot(t, θ1,  label="θ1", ylabel="rad", color=:blue)
         plot!(p2, t, θ2, label="θ2",            color=:red)
    hline!(p2, [x_eq[3]], linestyle=:dash, color=:blue,  label="θ1 target")
    hline!(p2, [x_eq[4]], linestyle=:dash, color=:red,   label="θ2 target")

    p3 = plot(t, dθ1, label="θ̇1", ylabel="rad/s", color=:blue)
         plot!(p3, t, dθ2, label="θ̇2",             color=:red)

    p4 = plot(t, dr,  label="ṙ (cart vel)", ylabel="m/s", color=:black)

    plt = plot(p1, p2, p3, p4,
               layout=(4,1), xlabel="t (s)",
               title=["Cart position" "Angles" "Angular velocities" "Cart velocity"],
               size=(900, 900), legend=:topright)
    display(plt)
    savefig(plt, "states.png")
    println("Saved states.png")
end

# ---------------------------------------------------------------------------
# Phase portraits
# ---------------------------------------------------------------------------

function plot_phase(sol, x_eq)
    _, _, _, θ1, θ2, dθ1, dθ2 = extract(sol)

    p1 = plot(θ1, dθ1, label="θ1 phase", xlabel="θ1 (rad)", ylabel="θ̇1 (rad/s)", color=:blue)
    scatter!(p1, [x_eq[3]], [0.0], marker=:star5, color=:black, label="target")

    p2 = plot(θ2, dθ2, label="θ2 phase", xlabel="θ2 (rad)", ylabel="θ̇2 (rad/s)", color=:red)
    scatter!(p2, [x_eq[4]], [0.0], marker=:star5, color=:black, label="target")

    plt = plot(p1, p2, layout=(1,2), size=(900, 400))
    display(plt)
    savefig(plt, "phase.png")
    println("Saved phase.png")
end

# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------

function animate_pendulum(sol, p; fps=30, filename="pendulum.gif")
    t, r, _, θ1, θ2, _, _ = extract(sol)
    l1, l2 = p.l1, p.l2

    # Subsample to target fps
    dt_anim = 1.0 / fps
    t_anim  = collect(t[1]:dt_anim:t[end])
    r_s  = [r[argmin(abs.(t .- ti))]  for ti in t_anim]
    θ1_s = [θ1[argmin(abs.(t .- ti))] for ti in t_anim]
    θ2_s = [θ2[argmin(abs.(t .- ti))] for ti in t_anim]

    lim = l1 + l2 + 0.5

    anim = @animate for i in eachindex(t_anim)
        rx  = r_s[i]
        m1x = rx + l1*sin(θ1_s[i]);  m1y = -l1*cos(θ1_s[i])
        m2x = m1x + l2*sin(θ2_s[i]); m2y = m1y - l2*cos(θ2_s[i])

        plot(legend=false, xlim=(-lim, lim), ylim=(-lim-0.5, lim),
             aspect_ratio=:equal, title="t = $(round(t_anim[i], digits=2)) s")

        # Track
        hline!([0.0], color=:gray, linestyle=:dash)
        # Cart
        plot!([rx-0.2, rx+0.2, rx+0.2, rx-0.2, rx-0.2],
              [0.05, 0.05, -0.05, -0.05, 0.05], color=:black, fill=true)
        # Rods
        plot!([rx, m1x], [0.0, m1y], color=:black, lw=2)
        plot!([m1x, m2x], [m1y, m2y], color=:black, lw=2)
        # Masses
        scatter!([m1x], [m1y], markersize=10, color=:blue)
        scatter!([m2x], [m2y], markersize=10, color=:red)
    end

    gif(anim, filename, fps=fps)
    println("Saved $filename")
end

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

plot_states(sol, x_eq)
plot_phase(sol, x_eq)
animate_pendulum(sol, p_phys)
