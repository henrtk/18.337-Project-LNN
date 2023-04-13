using LNNProject
using Test

@testset "LNNProject.jl" begin
    # Write your unit tests here. This is to ensure that code behaviour is correct
    # and that it does not change suddenly. If it does change, but this is expected,
    # change the tests. 
    @test LNNProject.test_fn(1,2) == 2*1
end
