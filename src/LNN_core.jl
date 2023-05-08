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

function _lagrangian_forward(re, p , qv)
    N = size(q_qdot, 1)÷2
    # Check if views work here later
    q = qv[1:N]
    v = qv[N+1:end]
    
    # Assuming single output for now, we treat L as a scalar, otherwise we would have to
    # do something integration-like for a lagrangian density
    function L(q, v, p)
        sum(re(p)([q;v]))
    end

    L_q  = (q, v, p) -> Zygote.gradient(x -> L(x, v, p), q)[1]
    L_v  = (q, v, p) -> Zygote.gradient(x -> L(q, x, p), v)[1]
    L_vq = (q, v, p) -> Zygote.jacobian(x -> L_v(x, v, p), q)[1]
    ∇ᵥ²L = (q, v, p) -> Zygote.hessian( x -> L(q, x, p), v)

    # pinv for stability and non-singularity
    ∇ᵥ²L⁻¹ = (q, v, ps) -> pinv(∇ᵥ²L(q, v, ps))
    # May want to solve linear system instead of inverting, but not possible if pinv.
    # to avoid singularities we may instead calculate for ∇²L-epsilon*I
    #eI = 2*eps()*I 
    v̇ = ∇ᵥ²L⁻¹(q,v,p)*(L_q(q,v,p) - L_vq(q,v,p) * v) 
    return vcat(v, v̇)
end

(lnn::LagrangianNN)(qv, p=lnn.p) = _lagrangian_forward(lnn.re,p, qv)

# Create wrapper for actual LagrangianNN layer

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
function (lnnL::NeuralLagrangian)(qv, p = lnnL.lnn.p, T = lnnL.tspan[2])
    function neural_Lagrangian_evolve!(vv̇, qv, p, t)
        v_vdot .= lnnL.lnn(qv, p)
    end
    prob = ODEProblem(neural_Lagrangian_evolve!, vv̇, (0, T), p)
    sensitivity = DiffEqFlux.InterpolatingAdjoint(autojacvec = false)
    solve(prob, Tsit5(), lnnL.args...; sensealg = sensitivity, lnnL.kwargs...)
end  
