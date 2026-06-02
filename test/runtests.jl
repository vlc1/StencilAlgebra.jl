using StencilAlgebra
using ScalarAlgebra
using LinearAlgebra
using Test

@testset "StencilAlgebra" begin

    @testset "AbstractField eltype" begin
        @test eltype(AbstractField{Float64}) === Float64
        @test eltype(AbstractField{Float32}) === Float32
    end

    @testset "FieldSym" begin
        fd = @inferred FieldSym{:u}()
        @test fd isa FieldSym{:u, Float64}
        @test fd isa AbstractField{Float64}
        @test eltype(fd) === Float64

        fd32 = @inferred FieldSym{:u, Float32}()
        @test fd32 isa FieldSym{:u, Float32}
        @test eltype(fd32) === Float32

        @test_throws ArgumentError FieldSym{:u, AbstractFloat}()
    end

    @testset "@field macro" begin
        @field u
        @test u isa FieldSym{:u, Float64}

        @field v Float32
        @test v isa FieldSym{:v, Float32}
    end

    @testset "Fill from AbstractScalar" begin
        sc = ScalarConst(3.0)
        f = @inferred Fill(sc)
        @test f isa Fill{Float64, ScalarConst{Float64}}
        @test f isa AbstractField{Float64}
        @test eltype(f) === Float64
        @test f.val === sc

        sym = ScalarSym{:a}()
        fs = @inferred Fill(sym)
        @test fs isa Fill{Float64}
        @test eltype(fs) === Float64
    end

    @testset "Fill from plain value" begin
        f = Fill(2.0)
        @test f isa Fill{Float64}
        @test eltype(f) === Float64

        f32 = Fill(Float32(1.0))
        @test f32 isa Fill{Float32}
        @test eltype(f32) === Float32
    end

    @testset "Fill rejects abstract T" begin
        @test isconcretetype(Float64)
        @test !isconcretetype(AbstractFloat)
    end

    @testset "AbstractStencil" begin
        @test AbstractStencil{Float64} isa Type
        @test isabstracttype(AbstractStencil)
    end

    @testset "FieldZero" begin
        fz = FieldZero(Float64)
        @test fz isa Fill{Bool, ScalarZero{Bool}}
        @test fz isa FieldZero{Bool}
        @test fz isa AbstractField{Bool}
        @test eltype(fz) === Bool

        fz_val = FieldZero(1.0)
        @test fz_val isa FieldZero{Bool}

        fz32 = FieldZero(Float32)
        @test fz32 isa FieldZero{Bool}
    end

    @testset "StencilOne" begin
        fo = StencilOne(Float64)
        @test fo isa StencilOne{Bool}
        @test fo isa AbstractStencil{Bool}
        @test eltype(fo) === Bool

        fo_val = StencilOne(1.0)
        @test fo_val isa StencilOne{Bool}

        d = LinearAlgebra.diag(StencilOne(Float64))
        @test d isa Fill{Bool, ScalarOne{Bool}}
        @test d isa AbstractField{Bool}

        @test_throws ArgumentError StencilOne{Float64}()
    end

    @testset "FieldCall" begin
        u = FieldSym{:u}()
        v = FieldSym{:v}()
        v32 = FieldSym{:v, Float32}()

        fc = @inferred FieldCall(+, (u, v))
        @test fc isa FieldCall
        @test fc isa AbstractField{Float64}
        @test eltype(fc) === Float64
        @test fc.fn === (+)
        @test fc.args === (u, v)

        fc_mixed = FieldCall(+, (u, v32))
        @test eltype(fc_mixed) === Float64

        fc_neg = @inferred -u
        @test fc_neg isa FieldCall
        @test eltype(fc_neg) === Float64

        fc_add = @inferred u + v
        @test fc_add isa FieldCall
        @test eltype(fc_add) === Float64

        fc_scale = @inferred u * 2.0
        @test fc_scale isa FieldCall
        @test eltype(fc_scale) === Float64

        s = FieldSym{:s, String}()
        @test_throws ArgumentError FieldCall(+, (s, s))
    end

    @testset "Un-dotted arithmetics" begin
        u = FieldSym{:u}()
        v = FieldSym{:v}()
        w = FieldSym{:w}()

        # binary ops between fields create FieldCall via broadcast machinery
        # (or via un-dotted operator overloads which do the same thing)
        bc_add = @inferred u + v
        @test bc_add isa FieldCall
        @test eltype(bc_add) === Float64
        @test bc_add.fn === (+)

        # un-dotted with scalar literal (lifted via asfield)
        bc_scale = @inferred u * 3.0
        @test bc_scale isa FieldCall
        @test eltype(bc_scale) === Float64

        # un-dotted from scalar side
        bc_rev = @inferred 2.0 * u
        @test bc_rev isa FieldCall
        @test eltype(bc_rev) === Float64

        # nested un-dotted ops (each level creates FieldCall)
        bc_nested = @inferred (u + v) * 0.5
        @test bc_nested isa FieldCall
        @test eltype(bc_nested) === Float64

        # type promotion in ops
        v32 = FieldSym{:v, Float32}()
        bc_mixed = @inferred u + v32
        @test bc_mixed isa FieldCall
        @test eltype(bc_mixed) === Float64

        # subtraction
        bc_sub = @inferred u - v
        @test bc_sub isa FieldCall
        @test bc_sub.fn === (-)

        # three-way operation
        bc_triple = @inferred u + v - w
        @test bc_triple isa FieldCall
    end

    @testset "Broadcasting" begin
        # exercise FieldStyle machinery defined in operators.jl

        @field u
        @field v
        @field w

        # Unary broadcast ops (element-wise functions)
        neg_u = @inferred .-u
        @test neg_u isa FieldCall
        @test eltype(neg_u) === Float64

        # Binary broadcast: field .+ field
        bc_add = @inferred u .+ v
        @test bc_add isa FieldCall
        @test eltype(bc_add) === Float64
        @test bc_add.fn === (+)

        # Binary broadcast: field .- field
        bc_sub = @inferred u .- v
        @test bc_sub isa FieldCall
        @test bc_sub.fn === (-)

        # Binary broadcast: field .* scalar
        bc_mul = @inferred u .* 2.0
        @test bc_mul isa FieldCall
        @test eltype(bc_mul) === Float64

        # Binary broadcast: scalar .* field
        bc_rev_mul = @inferred 3.0 .* u
        @test bc_rev_mul isa FieldCall
        @test eltype(bc_rev_mul) === Float64

        # Binary broadcast: field ./ scalar
        bc_div = @inferred u ./ 2.0
        @test bc_div isa FieldCall
        @test bc_div.fn === (/)

        # Nested broadcast ops
        bc_nested = @inferred (u .+ v) .* w
        @test bc_nested isa FieldCall
        @test eltype(bc_nested) === Float64

        # Broadcast with type promotion
        @field v32 Float32
        bc_mixed = @inferred u .+ v32
        @test bc_mixed isa FieldCall
        @test eltype(bc_mixed) === Float64

        # Three-way broadcast
        bc_triple = @inferred u .+ v .- w
        @test bc_triple isa FieldCall
    end

end
