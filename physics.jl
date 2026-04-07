using DifferentialEquations
#using ControlSystems
#using ForwardDiff
using LinearAlgebra

struct Parameters
    m1::Float64
    m2::Float64
    M::Float64
    l1::Float64
    l2::Float64
    g::Float64
    u::Float64
end

function doublependulum!(dx, x, p, _)
    r, dr, theta1, theta2, dtheta1, dtheta2 = x
    m1, m2, M, l1, l2, g, u = p.m1, p.m2, p.M, p.l1, p.l2, p.g, p.u

    # Mass matrix from Lagrangian (see LaTeX derivation)
    # Generalised coordinates: [r, theta1, theta2]
    mass_matrix = [
        M+m1+m2                  (m1+m2)*l1*cos(theta1)          m2*l2*cos(theta2)            ;
        (m1+m2)*l1*cos(theta1)   (m1+m2)*l1^2                    m2*l1*l2*cos(theta2-theta1)  ;
        m2*l2*cos(theta2)         m2*l1*l2*cos(theta2-theta1)    m2*l2^2                      ;    
    ]

    # RHS forcing vector
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

# --- Test simulation (no control) ---
# m1=1, m2=1, M=5, l1=1, l2=1, g=9.81, u=0
p  = Parameters(1.0, 1.0, 5.0, 1.0, 1.0, 9.81, 0.0)
x0 = [0.0, 0.0, π+0.1, π-0.1, 0.0, 0.0]   # near upright
tspan = (0.0, 10.0)
prob = ODEProblem(doublependulum!, x0, tspan, p)
sol  = solve(prob, Tsit5())
println("Solved $(length(sol.t)) time steps, t_end = $(sol.t[end])")
