# Swing-up via trajectory optimisation using TrajectoryOptimization.jl + Altro.jl
# Massless rods, point masses, no friction.
#
# Install:  ]add TrajectoryOptimization RobotDynamics Altro StaticArrays

using TrajectoryOptimization
using RobotDynamics
import Altro
using StaticArrays
using LinearAlgebra


# System parameters  
const m1_ = 1.0;  const m2_ = 1.0;  const M_  = 50.0
const l1_ = 1.0;  const l2_ = 1.0;  const g_  = 9.81
# State:   x = [r, ṙ, θ1, θ2, θ̇1, θ̇2]
# Control: u = [F]  (cart force)

struct DoublePendulumCart <: RobotDynamics.ContinuousDynamics end

RobotDynamics.state_dim(::DoublePendulumCart) = 6
RobotDynamics.control_dim(::DoublePendulumCart) = 1

function RobotDynamics.dynamics(::DoublePendulumCart, x, u)
    _, dr, theta1, theta2, dtheta1, dtheta2 = x
    F = u[1]

    m1, m2, M, l1, l2, g = m1_, m2_, M_, l1_, l2_, g_

    mass_matrix = @SMatrix [
        M+m1+m2                (m1+m2)*l1*cos(theta1)          m2*l2*cos(theta2)          ;
        (m1+m2)*l1*cos(theta1)    (m1+m2)*l1^2                 m2*l1*l2*cos(theta2-theta1)   ;
        m2*l2*cos(theta2)          m2*l1*l2*cos(theta2-theta1)         m2*l2^2                ;
    ]

    f = @SVector [
        (m1+m2)*l1*sin(theta1)*dtheta1^2 + m2*l2*sin(theta2)*dtheta2^2 + F,
        -m2*l1*l2*sin(theta1-theta2)*dtheta2^2 - (m1+m2)*g*l1*sin(theta1),
         m2*l1*l2*sin(theta1-theta2)*dtheta1^2 - m2*g*l2*sin(theta2),
    ]

    ddq = mass_matrix \ f   

    return @SVector [dr, ddq[1], dtheta1, dtheta2, ddq[2], ddq[3]]
end


function make_swingup_problem(x0, x_target; N=401, tf=36.0, u_max=40.0)
    model = DoublePendulumCart()
    n, m  = state_dim(model), control_dim(model)

    Q  = Diagonal(@SVector fill(1e-2, n))
    R  = Diagonal(@SVector fill(5e-2, m))
    Qf = Diagonal(@SVector [2000., 2000., 150000., 150000., 15000., 15000.])

    x0_s = SVector{n}(x0)
    xf_s = SVector{n}(x_target)

    cost      = LQRCost(Q, R, xf_s)
    cost_term = LQRCost(Qf, R, xf_s)
    obj       = Objective(cost, cost_term, N)

    # Constraints: control bounds. Keep the terminal target as a cost first;
    # hard terminal equality is often too strict for the initial solve.
    cons  = ConstraintList(n, m, N)
    u_bnd = BoundConstraint(n, m, u_min=SA[-u_max], u_max=SA[u_max])
    add_constraint!(cons, u_bnd, 1:N-1)

    prob = Problem(model, obj, x0_s, tf, xf=xf_s, constraints=cons)

    # Warm start: smooth trajectory from down to upright with modest cart motion.
    # A feasible-looking initial guess helps ALTRO avoid getting stuck early.
    dt = tf / (N - 1)
    u0 = [@SVector zeros(m) for _ in 1:N-1]
    x_init = map(1:N) do k
        s  = (k - 1) / (N - 1)           # normalised time in [0,1]
        α  = 0.5 * (1 - cos(π * s))      # smooth 0→1
        dα = 0.5 * π * sin(π * s) / tf   # derivative of α wrt t

        # Bias the cart with a small excursion so the optimizer has room to
        # generate angular momentum without fighting a perfectly fixed cart.
        cart_excursion = 0.5 * sin(π * s)
        r    = x0[1] + (x_target[1] - x0[1]) * α + cart_excursion
        dr   = (x_target[1] - x0[1]) * dα + 0.5 * π * cos(π * s) / tf
        θ1   = x0[3] + (x_target[3] - x0[3]) * α
        θ2   = x0[4] + (x_target[4] - x0[4]) * α
        dθ1  = (x_target[3] - x0[3]) * dα
        dθ2  = (x_target[4] - x0[4]) * dα
        SVector{n}(r, dr, θ1, θ2, dθ1, dθ2)
    end
    initial_controls!(prob, u0)
    initial_states!(prob, x_init)

    return prob
end
















#----------------------------------------------------------------------------------
x0       = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # stable equilibrium
x_target = [0.0, 0.0, π, π, 0.0, 0.0]   # both-up

prob   = make_swingup_problem(x0, x_target)
solver = Altro.ALTROSolver(prob)
Altro.set_options!(solver, verbose=1, iterations=1023, iterations_outer=50)
Altro.solve!(solver)

X = Altro.states(solver)
U = Altro.controls(solver)

println("\nSolver status: ", Altro.status(solver))
println("Final cost:    ", round(Altro.cost(solver), digits=3))
println("Final state:   ", round.(Vector(X[end]), digits=4))
println("Target:        ", round.(x_target, digits=4))
println("Terminal error:", round.(Vector(X[end]) .- x_target, digits=4))
println("Max control:   ", round(maximum(abs, [u[1] for u in U]), digits=3))

if Altro.status(solver) != Altro.SOLVE_SUCCEEDED
    @warn "Optimiser did not converge — trajectory may be unusable. Check terminal error before running sim.jl."
end
