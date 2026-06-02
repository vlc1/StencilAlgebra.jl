"""
    AbstractStencil{T}

Supertype for linear transformations of `AbstractField`s.
Applied by invoking `*`.
"""
abstract type AbstractStencil{T} end

Base.eltype(::Type{<:AbstractStencil{T}}) where {T} = T
Base.eltype(st::AbstractStencil) = eltype(typeof(st))

"""
    StencilOne{T}()
    StencilOne(U::Type) / StencilOne(x)

Type-level multiplicative identity for [`AbstractField`](@ref): the
pointwise-side analogue of scalar-side [`ScalarOne`](@ref), now reified as a
*stencil* (so it can serve as the neutral element of `*(stencil, field)`.

The parameter `T` is **bool-shaped** (`Bool` or `AbstractArray{Bool}`),
mirroring `ScalarOne`'s discipline so that promotion in surrounding arithmetic
recovers the value type without pinning a Stencil's coefficient eltype. The
second parameter `U` is the *value space* (e.g. `Float64`, `SMatrix{N,N,F}`)
recovered at materialize time via `one(U)`.

The outer constructors map any concrete value-space type `U` to its bool-
shape `T = _to_bool_shape(_unity_space(U))`, so
`StencilOne(Float64) === StencilOne{Bool, Float64}()` and
`eltype(StencilOne(Float64)) === Bool`. Materializes to `one(U)` —
e.g. `1.0` for `U = Float64`.

See also [`FieldZero`](@ref) for the additive identity.
"""
struct StencilOne{T} <: AbstractStencil{T}

    function StencilOne{T}() where {T}
        _assert_bool_shape(:StencilOne, T)
        applicable(one, T) || throw(ArgumentError(
            "StencilOne{T} requires `one(T)` to be defined (a " *
            "square-scalar shape); got T=$T"))
        new{T}()
    end
end

StencilOne(::Type{U}) where {U} =
       StencilOne{_to_bool_shape(_unity_space(U))}()
StencilOne(::U) where {U} = StencilOne(U)

LinearAlgebra.diag(::StencilOne{T}) where {T} = Fill(ScalarOne(T))

# for now
ScalarAlgebra.simplify(st::AbstractStencil) = st
