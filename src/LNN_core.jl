# For safety (??)
abstract type NeuralDELayer <: Function end
Flux.trainable(m::NeuralDELayer) = (m.p,)

"""
Docs (to come)
A Lagrangian neural network is a neural network that learns the underlying
Lagrangian of a system. The Lagrangian is a function of the system's state and its
time derivative, L(q, qdot), and thus needs 2n inputs, where n is the dimension of
the state space. The inputs are assumed to be ordered as [q, qdot].

"""
struct LagrangianNN{M, R, P}
    model::M
    re::R
    p::P 
    function LagrangianNN(model; p = nothing)
        _p, re = Flux.destructure(model)
        p = p === nothing ? _p : p
        return new{typeof(model), typeof(re), typeof(p)}(model, re, p)
    end
end
    
Flux.trainable(lnn::LagrangianNN) = (lnn.p,)

function _lagrangian_forward(re, p, q_qdot)
    N = size(q_qdot, 1)÷2
    qdot = @view q_qdot[N+1:end]
    # Assuming 1D for now, we treat L as a scalar, otherwise we would have to
    # do something integration-like
    L = x -> sum(re(p)(x))

    # There has to be a more efficient way to do this mathematically
    ∇_q∇ₚᵀL = Zygote.jacobian(q_qdot-> Zygote.gradient(L,q_qdot)[1][N+1:end], q_qdot)[1][1:N,N+1:end]

    ∇_qL = Zygote.gradient(L, q_qdot)[1][1:N]
    
    # Pseudo-inverse to avoid singularities
    ∇²L⁻¹ = pinv(Zygote.hessian(L, q_qdot)[1])
    
    qdotdot = ∇²L⁻¹*(∇_qL - ∇_q∇ₚᵀL * qdot) 
    return [qdot, qdotdot]
end

(lnn::LagrangianNN)(q_qdot, p = lnn.p) = _lagrangian_forward(lnn.re, p, q_qdot)

# Create wrapper for actual LagrangiaNN layer

struct NeuralLagrangian{M,P,RE,T,A,K} <: NeuralDELayer
    lnn::LagrangianNN
    p::P
    re::RE
    tspan::T
    args::A
    kwargs::K

    function NeuralLagrangian(model, tspan, p = nothing, args...; kwargs...)
        lnn = LagrangianNN(model, p=p)
        
        new{    
            typeof(lnn.model),
            typeof(lnn.p),
            typeof(lnn.re), 
            typeof(tspan), 
            typeof(args), 
            typeof(kwargs)
            }(lnn, lnn.p, lnn.re, tspan, args, kwargs)
        
    end

    # dispatch for the case where the user passes in a pre-constructed LagrangianNN
    function NeuralLagrangian(lnn::LagrangianNN, tspan, args...; kwargs...)
        new{    
            typeof(lnn.model),
            typeof(lnn.p),
            typeof(lnn.re), 
            typeof(tspan), 
            typeof(args), 
            typeof(kwargs)
            }(lnn, lnn.p, lnn.re, tspan, args, kwargs)
    end

end

# Define the output of the LNN when called
function (lnnL::NeuralLagrangian)(q_qdot, p = lnnL.lnn.p)
    function neural_Lagrangian_evolve!(qdot_qdotdot, q_qdot, p, t)
        qdot_qdotdot .= reshape(lnnL.lnn(q_qdot, p), sizeof(qdot_qdotdot))
    end
    prob = ODEProblem(neural_Lagrangian_evolve!, q_qdot, lnnL.tspan, p)
    sensitivity = DiffEqFlux.InterpolatingAdjoint(autojacvec = false)
    return solve(prob, Tsit5(), lnnL.args...; sensealg = sensitivity, lnnL.kwargs...)
end  

