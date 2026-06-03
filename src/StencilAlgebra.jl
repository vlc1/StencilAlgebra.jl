module StencilAlgebra

using LinearAlgebra,
      StaticArrays,
      ScalarAlgebra

using ScalarAlgebra: _assert_concrete,
                     _assert_bool_shape,
                     _to_bool_shape,
                     _unity_space,
#                     _value_space,
                     asscalar

export Offset,
       Shift,
       dim, pos,
       ô, ê₁, ê₂, ê₃, ê₄, ê₅, ê₆, ê₇, ê₈, ê₉,
       AbstractField,
       FieldSym,
       Fill,
       FieldZero,
       FieldCall,
       Shifted,
       @field,
       AbstractStencil,
       StencilOne

include("static.jl")
include("fields.jl")
include("stencils.jl")
include("operators.jl")
include("display.jl")

end # module
