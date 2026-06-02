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

export AbstractField,
       FieldSym,
       Fill,
       FieldZero,
       FieldCall,
       @field,
       AbstractStencil,
       StencilOne

include("fields.jl")
include("stencils.jl")
include("operators.jl")
include("display.jl")

end # module
