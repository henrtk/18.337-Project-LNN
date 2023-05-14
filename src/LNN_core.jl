
"""
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
        # if p is not set, set it to be the ps of the model
        p = p === nothing ? _p : p
        return new{typeof(model), typeof(re), typeof(p)}(model, re, p)
    end
end

"""
Defines the Right hand side of the ODE that is solved to generate the system's dynamics
"""
function _lagrangian_forward(qv, p , re)
    N = size(qv, 1)÷2
    # Check if views work here later
    q = @view qv[1:N]
    v = @view qv[N+1:end]
    model = re(p)
    # Assuming single output for now, we treat L as a scalar, otherwise we would have to
    # do something integration-like for a generalized lagrangian density with multiple outputs
    function L(q, v)
        sum(model([q;v]))
    end

    L_q  = (q, v) -> Zygote.gradient(x -> L(x, v), q)[1]
    L_v  = (q, v) -> Zygote.gradient(x -> L(q, x), v)[1]
    L_vq = (q, v) -> ForwardDiff.jacobian(x -> L_v(x, v), q)[1]
    ∇ᵥ²L = (q, v) -> Zygote.hessian(x -> L(q, x), v)

    # pinv for stability and singularity avoidance (pinv does not seem to explode close to singularities)
    ∇ᵥ²L⁻¹ = (q, v) -> pinv(∇ᵥ²L(q, v))
    # May want to solve linear system instead of inverting, but not possible if relying on pinv.
    # to avoid singularities we may instead calculate for ∇²L-epsilon*I with a linear system
    # eI = 2*eps()*I 
    v̇ = ∇ᵥ²L⁻¹(q,v)*(L_q(q,v) - L_vq(q,v) * v) 
    return vcat(v, v̇)
end

"""
Generates the LNN's prediction for the right hand side of the ODE. 
Assumes the form of the input vector to be [q ; v] where holds the position data
and v holds the respective speed data where d/dt q_i =: v_i.
"""
(lnn::LagrangianNN)(qv, p=lnn.p) = _lagrangian_forward(qv, p, lnn.re)


# Create wrapper for actual LagrangianNN layer

"""
NeuralLagrangian is a wrapper for the LagrangianNN layer that allows it to be used
as a Flux layer. It is meant to be a drop-in replacement for a Flux layer, but this is still WIP.
Holds the LagrangianNN, the parameters of the neural network, the range of time over which
the system is to be evolved, and any additional arguments to be passed to the ODE solver given by 
the user through the constructor.

When called:
    
(NL::NeuralLagrangian)(qv, forward_method = Euler(), p = NL.p, T = NL.tspan[2])
    Generates the LNN's prediction of the system's state at a given time T given the initial
    state qv and the parameters of the neural network p.

    variables:
        qv: the initial state of the system, assumed to be on 
            the form [q;v], where q holds he position data and
            v holds the respective speed data where 
            d/dt q_i =: v_i.
        forward_method: the method used to evolve the ODE 
            (default Euler; RHS evaluations are expensive and 
            higher order methods fail unexpectedly). 
            Beware that the adjoint method is not yet implemented 
            for NeuralLagrangian and that adaptive step size 
            methods may yield excessive runtimes when untrained as 
            the Neural Net often seems to yield an unstable ODE
            when untrained. The user may try VCAB3() or 
            AutoTsit5(Rosenbrock23()) if stiffness is an issue
        p: the parameters of the neural network
        T: the time at which to predict the state of the system
            
    returns:
        solve object containing the state of the system at time T
        and the time steps taken to get there
"""
struct NeuralLagrangian{M,P,RE,T,A,K}
    lnn::LagrangianNN
    p::P
    re::RE
    tspan::T
    args::A
    kwargs::K

    """Construct a NeuralLagrangian layer from a normal Neural Network assumed to be compatible with Flux.

    variables: 
        model: the neural network to be used as the LagrangianNN
        tspan: the range of time over which the system is to be evolved by default
        p: the parameters of the neural network
        args: any additional arguments to be passed to the ODE solver
        kwargs: any additional keyword arguments to be passed to the ODE solver"""
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
    """Dispatch for when the given model is already a LagrangianNN struct."""
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
"""
Generates the LNN's prediction of the system's state at a given time T given the initial
state qv and the parameters of the neural network p. 

Note: the derivative of the NeuralLagrangian was supposed to be automatically defined through
SciMLSensitivity's automatic sensitivity analysis, but this fails as the AD-packages it uses do
not seem to support triple nesting of AD, which is required for the LNN as formulated by Cranmer et al (2019).
Finite differences can be used instead.

variables:
    qv: the initial state of the system, assumed to be on 
        the form [q;v], where q holds he position data and
        v holds the respective speed data where d/dt q_i =: v_i.
    forward_method: the method used to evolve the ODE 
        (default Euler; RHS evaluations are expensive and 
        higher order methods fail unexpectedly). 
        Beware that the adjoint method is not yet implemented for
        NeuralLagrangian and that adaptive step size methods may 
        yield excessive runtimes when untrained as the Neural Net 
        often seems to yield an unstable ODE when untrained.
        The user may try VCAB3() or AutoTsit5(stiffalg = Rosenbrock23)
    p: the parameters of the neural network
    T: the time at which to predict the state of the system

returns:
    solve object containing the state of the system at time T and the time steps taken to get there
"""
function (NL::NeuralLagrangian)(qv, forward_method = Euler(), p = NL.lnn.p, T = NL.tspan[2], adaptive = false, kwargs...)
    function neural_Lagrangian_evolve!(vv̇, qv, p, t)
        vv̇ .= NL.lnn(qv, p)
    end
    prob = ODEProblem(neural_Lagrangian_evolve!, qv, (0, T), p)
    # This does not work: sensitivity = DiffEqFlux.InterpolatingAdjoint(autojacvec = false)
    solve(prob, forward_method, NL.args..., adaptive = adaptive; NL.kwargs...)
end  
