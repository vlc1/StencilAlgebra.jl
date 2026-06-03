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

    @testset "Offset" begin
        o1 = Offset{1, 2}()
        @test dim(o1) === 1
        @test pos(o1) === 2
        @test o1 isa Offset{1, 2}

        o2 = Offset{3, -5}()
        @test dim(o2) === 3
        @test pos(o2) === -5

        # Zero magnitude is rejected
        @test_throws ArgumentError Offset{1, 0}()

        # Negation
        o_neg = @inferred -(Offset{2, 3}())
        @test dim(o_neg) === 2
        @test pos(o_neg) === -3

        # Scalar multiplication
        o_scaled = 2 * Offset{1, 3}()
        @test dim(o_scaled) === 1
        @test pos(o_scaled) === 6

        o_scaled2 = Offset{2, -2}() * 3
        @test dim(o_scaled2) === 2
        @test pos(o_scaled2) === -6
    end

    @testset "Shift" begin
        # Zero shift (identity)
        shift_id = @inferred Shift(())
        @test shift_id isa Shift{Tuple{}}
        @test @inferred iszero(shift_id)

        # Basis symbols
        @test ô isa Shift{Tuple{}}
        @test @inferred iszero(ô)

        # Unit shifts
        @test ê₁ isa Shift
        @test !iszero(ê₁)

        @test ê₂ isa Shift
        @test ê₃ isa Shift

        # Addition of shifts
        s1 = @inferred (ê₁ + ê₂)
        @test !iszero(s1)
        @test length(s1.offsets) === 2

        # Same-dimension offsets combine (normalization)
        s2 = ê₁ + 2ê₁
        @test length(s2.offsets) === 1
        # Should have magnitude 3
        o = first(s2.offsets)
        @test pos(o) === 3

        # Zero offset is dropped
        s_zero = @inferred (ê₁ - ê₁)
        @test iszero(s_zero)

        # Negation
        s_neg = @inferred (-ê₁)
        @test length(s_neg.offsets) === 1
        o_neg = first(s_neg.offsets)
        @test pos(o_neg) === -1

        # Subtraction
        s_sub = @inferred (ê₂ - ê₁)
        @test !iszero(s_sub)

        # Scalar multiplication
        s_scaled = 2 * ê₁
        @test length(s_scaled.offsets) === 1
        o_s = first(s_scaled.offsets)
        @test pos(o_s) === 2

        s_scaled2 = ê₂ * 3
        @test length(s_scaled2.offsets) === 1
        o_s2 = first(s_scaled2.offsets)
        @test pos(o_s2) === 3

        # Dimension sorting
        s_sorted = ê₃ + ê₁ + ê₂
        @test length(s_sorted.offsets) === 3
        @test dim(s_sorted.offsets[1]) === 1
        @test dim(s_sorted.offsets[2]) === 2
        @test dim(s_sorted.offsets[3]) === 3

        # Complex expressions
        s_complex = 3ê₁ + 2ê₂ - ê₃
        @test length(s_complex.offsets) === 3
        @test pos(s_complex.offsets[1]) === 3
        @test pos(s_complex.offsets[2]) === 2
        @test pos(s_complex.offsets[3]) === -1
    end

    @testset "Shifted" begin
        @field u
        @field v Float32
        @field w

        # Zero shift is identity
        result_id = @inferred Shifted(ô, u)
        @test result_id === u
        @test result_id isa AbstractField{Float64}

        # Non-zero shift creates Shifted node
        s1 = @inferred Shifted(ê₁, u)
        @test s1 isa Shifted
        @test s1 isa AbstractField{Float64}
        @test eltype(s1) === Float64

        # Shift field accessor
        @test s1.shift === ê₁
        @test s1.term === u

        # Multiple offset shift
        s_multi = @inferred Shifted(ê₁ + ê₂, u)
        @test s_multi isa Shifted
        @test !iszero(s_multi.shift)
        @test length(s_multi.shift.offsets) === 2

        # Element type preservation
        s_v32 = @inferred Shifted(ê₁, v)
        @test eltype(s_v32) === Float32
        @test s_v32.term === v

        # Nested shifted (flattened by simplification rule)
        s_nested = @inferred Shifted(ê₂, Shifted(ê₁, u))
        @test s_nested isa Shifted
        @test eltype(s_nested) === Float64
        @test s_nested.shift === (ê₂ + ê₁)
        @test s_nested.term === u

        # Shifted of FieldCall
        u_plus_v = u + v
        s_call = @inferred Shifted(ê₃, u_plus_v)
        @test s_call isa Shifted
        @test s_call.term isa FieldCall
        @test eltype(s_call) === Float64

        # Shifted with negative offset
        s_neg = @inferred Shifted(-ê₁, u)
        @test s_neg isa Shifted
        @test !iszero(s_neg.shift)

        # Shifted with complex shift
        s_complex = @inferred Shifted(3ê₁ + 2ê₂, w)
        @test s_complex isa Shifted
        @test length(s_complex.shift.offsets) === 2
        @test eltype(s_complex) === Float64

        # Zero shift with different field types
        s_zero_sym = Shifted(ô, u)
        @test s_zero_sym === u

        s_zero_fill = Shifted(ô, Fill(2.0))
        @test s_zero_fill isa Fill

        s_zero_call = Shifted(ô, u + v)
        @test s_zero_call isa FieldCall
    end

    @testset "Indexing-as-shift sugar" begin
        @field u
        @field v Float32

        # u[shift] syntax
        s1 = @inferred u[ê₁]
        @test s1 isa Shifted
        @test eltype(s1) === Float64
        @test s1 === Shifted(ê₁, u)

        # Complex shift
        s_multi = @inferred u[3ê₁ + 2ê₂]
        @test s_multi isa Shifted
        @test s_multi === Shifted(3ê₁ + 2ê₂, u)

        # Zero shift (identity)
        s_zero = @inferred u[ô]
        @test s_zero === u

        # Negative offset
        s_neg = @inferred u[-ê₁]
        @test s_neg isa Shifted
        @test !iszero(s_neg.shift)

        # Different field types
        s_v32 = @inferred v[ê₂]
        @test eltype(s_v32) === Float32

        # Chained shifts: u[ê₁][ê₂] = u[ê₁ + ê₂]
        s_chained = u[ê₁][ê₂]
        s_combined = u[ê₁ + ê₂]
        @test s_chained.shift === s_combined.shift
    end

    @testset "Shifted simplification rules" begin
        @field u
        @field v Float32

        # Rule 1: Fill is spatially invariant (any shift = identity)
        fill = Fill(2.0)
        @test @inferred Shifted(ê₁, fill) === fill
        @test @inferred Shifted(ê₂, fill) === fill
        @test @inferred Shifted(3ê₁ + 2ê₂, fill) === fill

        # Rule 2: FieldZero is spatially invariant (any shift = identity)
        fz = FieldZero(Float64)
        @test @inferred Shifted(ê₁, fz) === fz
        @test @inferred Shifted(ê₂, fz) === fz
        @test @inferred Shifted(3ê₁ + 2ê₂, fz) === fz

        # Rule 3: Shifting a Shifted combines shifts
        s_nested = @inferred Shifted(ê₂, Shifted(ê₁, u))
        @test s_nested isa Shifted
        @test s_nested.shift === (ê₂ + ê₁)
        @test s_nested.term === u

        # Deep nesting: Shifted(s3, Shifted(s2, Shifted(s1, u)))
        s1 = Shifted(ê₁, u)
        s2 = Shifted(ê₂, s1)
        s3 = Shifted(ê₃, s2)
        @test s3.shift === (ê₃ + ê₂ + ê₁)
        @test s3.term === u

        # Combining with complex shifts
        s_complex = Shifted(3ê₁ - ê₂, Shifted(ê₂ + 2ê₃, u))
        @test s_complex.shift === (3ê₁ - ê₂ + ê₂ + 2ê₃)
        @test s_complex.term === u
    end

    @testset "Base.parent for Shifted" begin
        @field u

        s = Shifted(ê₁, u)
        @test parent(s) === u
        @test parent(s) === s.term

        # Nested: parent unwraps one level
        s_nested = Shifted(ê₂, Shifted(ê₁, u))
        @test parent(s_nested) === u
    end

end
