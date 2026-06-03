"""
    AbstractField{T}

Supertype of every symbolic grid expression. An `AbstractField{T}` behaves
like a dimension- and size-less array whose `eltype` is `T`: its grid rank
`N` is unknown until it is materialized against concrete arrays, but its
element type `T` (the value each cell will hold once materialized) is fixed
at construction. Field analogue of [`AbstractScalar`](@ref).
"""
abstract type AbstractField{T} end

Base.eltype(::Type{<:AbstractField{T}}) where {T} = T
Base.eltype(fd::AbstractField) = eltype(typeof(fd))

"""
    FieldSym{S, T}()

Placeholder for a discrete field named `S` (a `Symbol`) whose cells hold
values of the concrete type `T` (default `Float64`). Substituted with an
`AbstractArray` at `materialize` and indexed per cell. Field analogue of
[`ScalarSym`](@ref).
"""
struct FieldSym{S, T} <: AbstractField{T}

    function FieldSym{S, T}() where {S, T}
        _assert_concrete(:FieldSym, T)
        new{S, T}()
    end
end

FieldSym{S}() where {S} = FieldSym{S, Float64}()

"""
    @field name [T = Float64]

Bind `name` to `FieldSym{:name, T}()`. `@field x` ≡
`x = FieldSym{:x, Float64}()`; `@field x Float32` ≡
`x = FieldSym{:x, Float32}()`.
"""
macro field(name, T = :Float64)
    name isa Symbol ||
        throw(ArgumentError("@field expects a variable name, got `$(name)`"))
    :($(esc(name)) = $FieldSym{$(QuoteNode(name)), $(esc(T))}())
end

"""
    Fill{T} <: AbstractField{T}

Broadcast-to-grid bridge from scalar-land to field-land: wraps a single value
(a literal or an [`AbstractScalar`](@ref)) and presents it as a spatially-
invariant `AbstractField`.
"""
struct Fill{T,S<:AbstractScalar{T}} <: AbstractField{T}
    val::S

    function Fill{T}(val::S) where {T,S<:AbstractScalar{T}}
        _assert_concrete(:Fill, T)
        new{T,S}(val)
    end
end

Fill(val::AbstractScalar) = Fill{eltype(val)}(val)
Fill(val) = Fill(asscalar(val))

"""
    FieldZero{T} = Fill{ScalarZero{T}}
    FieldZero(T::Type) / FieldZero(x)

Type-level additive identity for [`AbstractField`](@ref), defined as the `Fill`
of a scalar-side [`ScalarZero`](@ref). The parameter `T` is bool-shaped (`Bool`
or `AbstractArray{Bool}`), mirroring `ScalarZero`'s discipline; the outer
constructors map any concrete value-space type to its bool shape, so
`FieldZero(Float64) === Fill(ScalarZero{Bool}())` and
`eltype(FieldZero(Float64)) === Bool` (the `Fill{<:AbstractScalar}`
specialization reports `eltype(ScalarZero{Bool}) === Bool`; the `Float64` is
the *input* the bool-shape ctor consumes, recovered by promotion in surrounding
arithmetic).

Materializes to a broadcast of `zero(T)` (Bool false, etc.) — promotion in
surrounding arithmetic recovers the cell type, exactly as for
[`ScalarZero`](@ref) in scalar-land.
"""
const FieldZero{T} = Fill{T, ScalarZero{T}}

FieldZero(T::Type)       = Fill(ScalarZero(T))
FieldZero(::T) where {T} = Fill(ScalarZero(T))

"""
    FieldCall(fn, args::Tuple{Vararg{AbstractField}})

Internal node applying `fn` to `args` component-wise. The element type `T =
Base.promote_op(fn, eltype.(args)...)` is computed **at construction**; a
`Union{}` result (e.g. genuine `SVector` inhomogeneity) is an unconstructable
term and throws.
"""
struct FieldCall{F, A<:Tuple{Vararg{AbstractField}}, T} <: AbstractField{T}
    fn::F
    args::A

    FieldCall{F, A, T}(fn::F, args::A) where {F, A<:Tuple{Vararg{AbstractField}}, T} =
        new{F, A, T}(fn, args)
end

function FieldCall(fn::F, args::A) where {F, A<:Tuple{Vararg{AbstractField}}}
    T = Base.promote_op(fn, map(eltype, args)...)
    T === Union{} && throw(ArgumentError(
        "unconstructable FieldCall: $(fn) over eltypes $(map(eltype, args)) has no " *
        "result type (Base.promote_op returned Union{})"))
    FieldCall{F, A, T}(fn, args)
end

"""
    Shifted(shift::Shift, term::AbstractField)

A `term` read at the lattice `shift`. The element type is unchanged
(`eltype(term)`); the zero shift `ô` is the identity (returns `term`).
"""
struct Shifted{S<:Shift, T, U<:AbstractField{T}} <: AbstractField{T}
    shift::S
    term::U

    Shifted{S, T, U}(shift::S, term::U) where {S, T, U<:AbstractField{T}} =
        new{S, T, U}(shift, term)
end

Base.parent(this::Shifted) = this.term

# Zero shift with spatially invariant term
Shifted(::Shift{Tuple{}}, term::Union{Fill, FieldZero}) = term

# Zero shift is always identity
Shifted(::Shift{Tuple{}}, term::AbstractField) = term

# Spatially invariant terms return unchanged
Shifted(::Shift, term::Union{Fill, FieldZero}) = term

# Shifting a Shifted combines shifts
Shifted(shift::Shift, term::Shifted) = Shifted(shift + term.shift, parent(term))

# General case
Shifted(shift::Shift, term::AbstractField{T}) where {T} =
    Shifted{typeof(shift), T, typeof(term)}(shift, term)

asfield(fd::AbstractField) = fd
asfield(sc::AbstractScalar) = Fill(sc)
asfield(x) = Fill(ScalarConst(x))

# for now
ScalarAlgebra.simplify(fd::AbstractField) = fd
