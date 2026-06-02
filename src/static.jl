"""
    Offset{D, Z}

A single offset of magnitude `Z` along mesh dimension `D` (both
compile-time `Int`s). Singleton.
"""
struct Offset{D, Z}

    function Offset{D, Z}() where {D, Z}
        Z == 0 &&
            throw(ArgumentError("Offset with zero magnitude is not allowed."))
        new{D, Z}()
    end
end

dim(::Type{Offset{D, Z}}) where {D, Z}  = D
dim(::T) where {T <: Offset} = dim(T)

pos(::Type{Offset{D, Z}}) where {D, Z} = Z
pos(::T) where {T <: Offset} = pos(T)

# arithmetic
Base.:-(::Offset{D, Z}) where {D, Z} = Offset{D, -Z}()
Base.:*(k::Integer, ::Offset{D, Z}) where {D, Z} = Offset{D, k * Z}()
Base.:*(::Offset{D, Z}, k::Integer) where {D, Z} = Offset{D, k * Z}()

_insert(x::Offset, ::Tuple{}) = (x,)
function _insert(x::Offset{D₁, Z₁},
        (y, args...)::Tuple{Offset{D₂, Z₂}, Vararg{Offset}}) where {D₁, Z₁, D₂, Z₂}
    if D₁ == D₂
        Z = Z₁ + Z₂
        return iszero(Z) ? args : (Offset{D₁, Z}(), args...)
    elseif D₁ < D₂
        return (x, y, args...)
    else
        return (y, _insert(x, args)...)
    end
end

_normalize(::Tuple{}) = ()
_normalize((x, args...)::Tuple{Offset, Vararg{Offset}}) =
    _insert(x, _normalize(args))

"""
    Shift{P<:Tuple{Vararg{Offset}}}

A normalized collection of [`Offset`](@ref)s.  Invariants (enforced by the
constructor): offsets are sorted ascending by dimension, no two offsets share a
dimension (same-dimension offsets are summed), and no offset has zero offset
(dropped). The empty shift `Shift{Tuple{}}()` is the identity.

Construct from offsets, or via the basis symbols `ô`, `ê₁`, ..., `ê₉` and the
`+`/`-`/`*Int` algebra:

```julia
3ê₁ + ê₂ # Shift{Tuple{Offset{1, 3}, Offset{2, 1}}}
```
"""
struct Shift{T<:Tuple{Vararg{Offset}}}
    offsets::T

    function Shift(input::Tuple{Vararg{Offset}})
        output = _normalize(input)
        new{typeof(output)}(output)
    end
end

# arithmetic
Base.:+(a::Shift, b::Shift) = Shift((a.offsets..., b.offsets...))
Base.:-(a::Shift) = Shift(map(-, a.offsets))
Base.:-(a::Shift, b::Shift)  = a + (-b)
Base.:*(k::Integer, sh::Shift) = Shift(map(o -> k * o, sh.offsets))
Base.:*(sh::Shift, k::Integer) = Shift(map(o -> o * k, sh.offsets))

Base.iszero(::Shift{Tuple{}}) = true
Base.iszero(::Shift) = false

const _SUBSCRIPTS = ('₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉')

_subscript(n::Integer) = join(_SUBSCRIPTS[d - '0' + 1] for d in string(n))
_basis_symbol(D::Integer) = string('ê', _subscript(D))

function Base.show(io::IO, sh::Shift)
    if isempty(sh.offsets)
        print(io, "ô")
        return
    end
    first = true
    for offset in sh.offsets
        Z = pos(offset)
        if first
            Z < 0 && print(io, "-")
            first = false
        else
            print(io, Z < 0 ? " - " : " + ")
        end
        a = abs(Z)
        a == 1 || print(io, a)
        print(io, _basis_symbol(dim(offset)))
    end
end

"""
    ô, ê₁, ..., ê₉

Predefined [`Shift`](@ref) constants: `ô` is the zero shift (identity), and
`êᵢ` is the unit offset along axis `i`. Combine them with the `+`/`-`/`*Int`
algebra to write lattice offsets, e.g. `-2ê₁`, `3ê₁ + ê₂`.
"""
const ô  = Shift(())
const ê₁ = Shift((Offset{1, 1}(),))
const ê₂ = Shift((Offset{2, 1}(),))
const ê₃ = Shift((Offset{3, 1}(),))
const ê₄ = Shift((Offset{4, 1}(),))
const ê₅ = Shift((Offset{5, 1}(),))
const ê₆ = Shift((Offset{6, 1}(),))
const ê₇ = Shift((Offset{7, 1}(),))
const ê₈ = Shift((Offset{8, 1}(),))
const ê₉ = Shift((Offset{9, 1}(),))
