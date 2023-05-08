module LNNProject

using LinearAlgebra, DiffEqBase, Zygote, Flux, DiffEqFlux,
    DifferentialEquations, ReverseDiff, ForwardDiff

# Write your package code here.
include("LNN_core.jl")

export LagrangianNN, NeuralLagrangian

end # module
