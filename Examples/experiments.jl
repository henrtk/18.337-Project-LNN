
using DiffEqFlux, LNNProject, Plots, DifferentialEquations, Flux
using Optimization, ComponentArrays, OptimizationOptimJL, Random

function analytical_fn(du,u, p, t=0)
    t1, t2, w1, w2 = u[1], u[2], u[3], u[4]
    m1, m2, l1, l2, g = p[1], p[2], p[3], p[4], p[5]
    a1 = (l2 / l1) * (m2 / (m1 + m2)) * cos(t1 - t2)
    a2 = (l1 / l2) * cos(t1 - t2)
    f1 = -(l2 / l1) * (m2 / (m1 + m2)) * (w2^2) * sin(t1 - t2) - (g / l1) * sin(t1)
    f2 = (l1 / l2) * (w1^2) * sin(t1 - t2) - (g / l2) * sin(t2)
    g1 = (f1 - a1 * f2) / (1 - a1 * a2)
    g2 = (f2 - a2 * f1) / (1 - a1 * a2)
    du .=  [w1, w2, g1, g2]
end

ps2 =  [1, 1, 1, 1, 9.81]
init_state = [Float32(0.1), Float32(0.1), Float32(0.0), Float32(0.0)]
tsteps = 0:0.5:1


sol2 = solve(ODEProblem(analytical_fn, init_state, (0.0, 1.0),ps2), Tsit5(), saveat=tsteps)
ode_data = sol2(tsteps)
plot(sol2, label="Analytical")

model = Chain(
    Dense(4, 20, tanh),        
    Dense(20, 4)
)

model2 = Chain(
    Dense(4, 20, tanh),        
    Dense(20, 4)
)

tspan =(Float32(0.0),Float32(1.0))

lnn = NeuralLagrangian(model, tspan, saveat=tsteps)
NODE = NeuralODE(model2, tspan, sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()), saveat=tsteps)
NODE.p

C = p -> (NODE(init_state, p, 1.0)-ode_data)^2
ReverseDiff.gradient(C, NODE.p)

                                       #callback(result_neuralode.u, loss_neuralode(result_neuralode.u)...; doplot=true)