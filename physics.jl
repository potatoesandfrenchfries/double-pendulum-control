using DifferentialEquations
using ControlSystems
using ForwardDiff

struct Parameters
    m1::Float64
    m2::Float64
    l1::Float64
    l2::Float64
    g::Float64
end


function doublependulum!(dx,x,p,t) 
    theta1, theta2, dtheta1, dtheta2 = x

    m1, m2, l1, l2, g = p.m1, p.m2, p.l1, p.l2, p.g

    A = (m1+m2)*(l1^2)
    B = (m2*l1*l2*cos(theta2-theta1))
    C = (m2*l1*l2*(dtheta2^2)*sin(theta1-theta2)) + (m1+m2)*g*l1*sin(theta1)

    D = (m2*l1*l2*cos(theta2-theta1))
    E = (m2*l2^2)
    F = (-m2*l1*l2*(dtheta1^2)*sin(theta1-theta2)) + m2*g*l2*sin(theta2)

    ddtheta1 = (-C*E + B*F)/(A*E - B*D)
    ddtheta2 = (-A*F + C*D)/(A*E - B*D)

    dx[1] = dtheta1
    dx[2] = dtheta2
    dx[3] = ddtheta1
    dx[4] = ddtheta2
end