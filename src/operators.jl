# un-dotted arithmetic for AbstractField
Base.:+(a::AbstractField) = a
Base.:-(a::AbstractField) = FieldCall(-, (a,))

for op in (:+, :-)
    @eval Base.$op(a::AbstractField, b) = FieldCall($op, (a, asfield(b)))
    @eval Base.$op(a, b::AbstractField) = FieldCall($op, (asfield(a), b))
    @eval Base.$op(a::AbstractField, b::AbstractField) = FieldCall($op, (a, b))
end

Base.:*(t::AbstractField, c) = FieldCall(*, (t, asfield(c)))
Base.:*(c, t::AbstractField) = FieldCall(*, (asfield(c), t))
Base.:*(::AbstractField, ::AbstractField) = throw(ArgumentError(
    "Use `a .* b` instead of `a * b` for field multiplication."))

Base.:/(t::AbstractField, c) = FieldCall(/, (t, asfield(c)))
Base.:/(::AbstractField, ::AbstractField) = throw(ArgumentError(
    "Use `a ./ b` instead of `a / b` for field division."))

Base.:\(c, t::AbstractField) = FieldCall(\, (t, asfield(c)))
Base.:\(::AbstractField, ::AbstractField) = throw(ArgumentError(
    "Use `a .\\ b` instead of `a \\ b` for field division."))

Base.:^(t::AbstractField, c) = FieldCall(^, (t, asfield(c)))
Base.:^(::AbstractField, ::AbstractField) = throw(ArgumentError(
    "Use `a .^ b` instead of `a ^ b` for field exponentiation."))

# indexing-as-shift sugar
Base.getindex(term::AbstractField, shift::Shift) = Shifted(shift, term)

# broadcasting
struct FieldStyle <: Base.Broadcast.BroadcastStyle end

Base.BroadcastStyle(::Type{<:AbstractField}) = FieldStyle()

# FieldStyle absorbs everything we mix with.
Base.BroadcastStyle(::FieldStyle, ::FieldStyle) = FieldStyle()
Base.BroadcastStyle(::FieldStyle, ::Base.Broadcast.AbstractArrayStyle) = FieldStyle()
Base.BroadcastStyle(::Base.Broadcast.AbstractArrayStyle, ::FieldStyle) = FieldStyle()

# do not Ref-wrap our types in the broadcast machinery
Base.broadcastable(x::AbstractField) = x

# bypass `combine_axes` for our styles
Base.Broadcast.instantiate(bc::Base.Broadcast.Broadcasted{FieldStyle}) = bc

# materialize a Broadcasted{FieldStyle} into a Field tree. Nested Broadcasted
# args are recursively materialized; we intentionally do NOT
# `Broadcast.flatten` — each `.op` produces one Field node so the symbolic tree
# reflects the user's syntax (and the simplify rules see it that way).
function Base.copy(bc::Base.Broadcast.Broadcasted{FieldStyle})
    args = map(_pointwise_arg, bc.args)
    FieldCall(bc.f, args)
end

_pointwise_arg(b::Base.Broadcast.Broadcasted) = copy(b)
_pointwise_arg(x) = asfield(x)
