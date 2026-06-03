const _INFIX = (:+, :-, :*, :/, :\, :^)

Base.show(io::IO, t::AbstractField) = _show(io, simplify(t))

_show(io::IO, ::FieldSym{S}) where {S} = print(io, S)
_show(io::IO, ::FieldZero) = print(io, "O")
_show(io::IO, f::Fill) = show(io, f.val)

function _show(io::IO, s::Shifted)
    _show(io, s.term)
    print(io, '[', s.shift, ']')
end

function _show(io::IO, t::FieldCall)
    op, args = nameof(t.fn), t.args
    if length(args) == 2 && op in _INFIX
        print(io, '(')
        _show(io, args[1])
        print(io, ' ', op, ' ')
        _show(io, args[2])
        print(io, ')')
    elseif length(args) == 1 && op === :-
        print(io, '-')
        _show(io, args[1])
    else
        print(io, op, '(')
        for (i, a) in enumerate(args)
            i == 1 || print(io, ", ")
            _show(io, a)
        end
        print(io, ')')
    end
end

Base.show(io::IO, t::AbstractStencil) = _show(io, simplify(t))

_show(io::IO, ::StencilOne) = print(io, 'I')
