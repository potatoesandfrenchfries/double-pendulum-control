using DifferentialEquations
using LinearAlgebra
using ForwardDiff
using ControlSystems

struct Parameters
    m1::Float64
    m2::Float64
    M::Float64
    l1::Float64
    l2::Float64
    g::Float64
    control::Function   # (x, t) -> u
end
# x = [r, dr, theta1, theta2, dtheta1, dtheta2]
function doublependulum!(dx, x, p, t)
    r, dr, theta1, theta2, dtheta1, dtheta2 = x
    m1, m2, M, l1, l2, g = p.m1, p.m2, p.M, p.l1, p.l2, p.g
    u = p.control(x, t)

    mass_matrix = [
        M+m1+m2                (m1+m2)*l1*cos(theta1)        m2*l2*cos(theta2)          ;
        (m1+m2)*l1*cos(theta1) (m1+m2)*l1^2                  m2*l1*l2*cos(theta2-theta1);
        m2*l2*cos(theta2)       m2*l1*l2*cos(theta2-theta1)  m2*l2^2                    ;
    ]

    f = [
        (m1+m2)*l1*sin(theta1)*dtheta1^2 + m2*l2*sin(theta2)*dtheta2^2 + u;
        -m2*l1*l2*sin(theta1-theta2)*dtheta2^2 - (m1+m2)*g*l1*sin(theta1) ;
         m2*l1*l2*sin(theta1-theta2)*dtheta1^2 - m2*g*l2*sin(theta2)      ;
    ]

    # Cramer's Rule
    D = det(mass_matrix)
    ddr      = det([f          mass_matrix[:,2]  mass_matrix[:,3]]) / D
    ddtheta1 = det([mass_matrix[:,1] f           mass_matrix[:,3]]) / D
    ddtheta2 = det([mass_matrix[:,1] mass_matrix[:,2]  f         ]) / D

    dx[1] = dr
    dx[2] = ddr
    dx[3] = dtheta1
    dx[4] = dtheta2
    dx[5] = ddtheta1
    dx[6] = ddtheta2
end

# Control thingy

no_control = (x, t) -> 0.0


# Linearisation 
function linearise(p::Parameters, x_eq::Vector{Float64})
    # ForwardDiff requires a single array argument, so u is fixed here
    function F_state(x)
        dx = zeros(eltype(x), 6)
        doublependulum!(dx, x, Parameters(p.m1, p.m2, p.M, p.l1, p.l2, p.g, no_control), 0.0)
        return dx
    end

    # Describing the dynamics at x_eq
    function F_input(u_vec)
        dx = zeros(eltype(u_vec), 6)
        ctrl = (_, _) -> u_vec[1]
        doublependulum!(dx, x_eq, Parameters(p.m1, p.m2, p.M, p.l1, p.l2, p.g, ctrl), 0.0)
        return dx
    end
    
    #   A is the 6×6 Jacobian wrt state
    #   B is the 6×1 Jacobian wrt input
    A = ForwardDiff.jacobian(F_state, x_eq)  
    B = ForwardDiff.jacobian(F_input, [0.0])  

    return A, B
end

# The three unstable equlibria
x_up_up   = [0.0, 0.0, π, π, 0.0, 0.0]
x_up_down = [0.0, 0.0, π, 0.0, 0.0, 0.0]
x_down_up = [0.0, 0.0, 0.0, π, 0.0, 0.0]

# LQR - Trying to minimise cost function
# Q penalises state error, R penalises control effort.
# Larger Q[i,i] → faster correction of state i.
# Larger R      → less aggressive control (less force used).
# u = -K (x*)
# Ref notes for the clearer notes

function make_lqr(A, B, Q, R, x_eq)
    K = lqr(A, B, Q, R)        # 1×6 gain matrix
    return (x, _) -> -dot(K, x - x_eq)
end











#-------------------------------------------------------------------------

# Tests
p_phys = Parameters(1.0, 1.0, 5.0, 1.0, 1.0, 9.81, no_control)

# Q: penalise [r, ṙ, θ1, θ2, θ̇1, θ̇2] errors
# R: penalise control effort
Q = diagm([1.0, 1.0, 10.0, 10.0, 1.0, 1.0])
R = [0.1;;]   # 1×1 matrix

# Linearise and compute LQR gains for each equilibrium
A_uu, B_uu = linearise(p_phys, x_up_up)
A_ud, B_ud = linearise(p_phys, x_up_down)
A_du, B_du = linearise(p_phys, x_down_up)

ctrl_uu = make_lqr(A_uu, B_uu, Q, R, x_up_up)
ctrl_ud = make_lqr(A_ud, B_ud, Q, R, x_up_down)
ctrl_du = make_lqr(A_du, B_du, Q, R, x_down_up)

# Simulate LQR stabilisation from small perturbation near both-up
p_lqr = Parameters(p_phys.m1, p_phys.m2, p_phys.M, p_phys.l1, p_phys.l2, p_phys.g, ctrl_uu)
x0    = x_up_up + [0.0, 0.0, 0.05, 0.05, 0.0, 0.0]
prob  = ODEProblem(doublependulum!, x0, (0.0, 10.0), p_lqr)
sol   = solve(prob, Tsit5())
println("LQR both-up: solved $(length(sol.t)) steps")
println("Final state error: $(round.(sol.u[end] - x_up_up, digits=4))")
