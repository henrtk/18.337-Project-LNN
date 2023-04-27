
using DiffEqFlux, Plots, DifferentialEquations, Flux, Zygote,
   SciMLSensitivity, Plots, ReverseDiff, ForwardDiff

# Training data
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

ODE_params =  [1, 1, 1, 1, 9.81]
init_state = [Float32(0.1), Float32(0.1), Float32(0.0), Float32(0.0)]
tsteps = 0:0.1:1


Analytical_sol = solve(ODEProblem(analytical_fn, init_state, (0.0, 1.0), ODE_params), Tsit5(), saveat=tsteps)
ode_data = Array(Analytical_sol)

plot(sol2, label="Analytical")

# Neural ODE vvvvvv

NN_model = Chain(
    Dense(4, 40, tanh),        
    Dense(40, 4)
)

NODE_params, recreate_NODE_NN_from_NParams = Flux.destructure(model2)

function NODE_f(du, u, p, t)
    du .= recreate_NODE_NN_from_NParams(p)(u)
end

NODEprob = ODEProblem(NODE_f, init_state, (0.0, 5.0), )
solve(NODEprob, Tsit5(), saveat=tsteps, sensealg=InterpolatingAdjoint(),p=NODE_params)

function lossNODE(p)
    pred = solve(NODEprob, Tsit5(), saveat=tsteps, sensealg=InterpolatingAdjoint(autojacvec=false),p=p)
    loss = sum(abs2, ode_data .- Array(pred))
end

# Get gradient!
Zygote.gradient(lossNODE, NODE_params)

# Train!
opt = ADAM(0.1)


for i in 1:100
    grads = Zygote.gradient(lossNODE, NODE_params)[1]
    Flux.update!(opt, NODE_params, grads)
    if i%10 == 0
        @show lossNODE(NODE_params)
    end
end

plot(solve(NODEprob, Tsit5(), saveat=tsteps,p=NODE_params), label="NODE")