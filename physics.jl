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

no_control(x, t) = 0.0


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
# u = -K (x-x*)
# Ref notes for the clearer notes

function make_lqr(A, B, Q, R, x_eq)
    K = lqr(A, B, Q, R)        # 1×6 gain matrix
    return (x, _) -> -dot(K, x - x_eq)
end

# Energy functions 
function potential_energy(x, p)
    _, _, theta1, theta2, _, _ = x
    return -(p.m1 + p.m2)*p.g*p.l1*cos(theta1) - p.m2*p.g*p.l2*cos(theta2)
end

function kinetic_energy(x, p)
    _, dr, theta1, theta2, dtheta1, dtheta2 = x
    m1, m2, l1, l2 = p.m1, p.m2, p.l1, p.l2
    T  = 0.5*(p.M + m1 + m2)*dr^2
    T += 0.5*(m1 + m2)*l1^2*dtheta1^2
    T += 0.5*m2*l2^2*dtheta2^2
    T += (m1 + m2)*l1*cos(theta1)*dr*dtheta1
    T += m2*l2*cos(theta2)*dr*dtheta2
    T += m2*l1*l2*cos(theta2 - theta1)*dtheta1*dtheta2
    return T
end

total_energy(x, p) = kinetic_energy(x, p) + potential_energy(x, p)

# Model to use a swing up, and then switch to LQR near equilibrium


# u = clamp(γ · ṙ · (E − E_ref), −u_max, u_max)

# When E < E_ref: push with dr to add energy.
# When E > E_ref: push against dr to decrease energy.
# E_ref is the potential energy at the target equilibrium (zero velocity).

function make_swingup(p::Parameters, x_eq::Vector{Float64}; gamma=10.0, u_max=20.0)
    E_ref = potential_energy(x_eq, p)   # target energy (KE=0 at equilibrium)
    return function (x, _)
        E  = total_energy(x, p)
        dr = x[2]
        clamp(gamma * dr * (E - E_ref), -u_max, u_max)
    end
end

# ---------------------------------------------------------------------------
# Switching controller
# ---------------------------------------------------------------------------
# Use swing-up until within ε of the equilibrium, then hand off to LQR.
# Angle differences are wrapped to [-π, π] before computing distance.

function make_switching(swingup::Function, lqr_ctrl::Function,
                        x_eq::Vector{Float64}; epsilon=0.1)
    return function (x, t)
        delta = x - x_eq
        delta[3] = mod(delta[3] + π, 2π) - π   # wrap θ1 error
        delta[4] = mod(delta[4] + π, 2π) - π   # wrap θ2 error
        norm(delta) < epsilon ? lqr_ctrl(x, t) : swingup(x, t)
    end
end











#-------------------------------------------------------------------------
# Tests
#-------------------------------------------------------------------------

p_phys = Parameters(1.0, 1.0, 50.0, 1.0, 1.0, 9.81, no_control)

# Q weights: [r, ṙ, θ1, θ2, θ̇1, θ̇2]
Q = diagm([1.0, 1.0, 100.0, 100.0, 10.0, 10.0])
R = [0.1;;]

# Start from hanging down with a tiny nudge
x0 = [0.0, 0.0, 0.02, 0.02, 0.0, 0.0]

for (label, x_eq) in [
        ("both-up",  x_up_up),
        ("up-down",  x_up_down),
        ("down-up",  x_down_up),
    ]

    A, B     = linearise(p_phys, x_eq)
    lqr_ctrl = make_lqr(A, B, Q, R, x_eq)
    swingup  = make_swingup(p_phys, x_eq)
    ctrl     = make_switching(swingup, lqr_ctrl, x_eq)

    p_ctrl = Parameters(p_phys.m1, p_phys.m2, p_phys.M,
                        p_phys.l1, p_phys.l2, p_phys.g, ctrl)

    prob = ODEProblem(doublependulum!, x0, (0.0, 30.0), p_ctrl)
    sol  = solve(prob, Tsit5(), abstol=1e-8, reltol=1e-8)

    δ = sol.u[end] - x_eq
    δ[3] = mod(δ[3] + π, 2π) - π
    δ[4] = mod(δ[4] + π, 2π) - π
    println("[$label]  steps=$(length(sol.t))  final error=$(round.(δ, digits=4))")
end
