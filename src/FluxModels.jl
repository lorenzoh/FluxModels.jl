module FluxModels

using DocStringExtensions
using Flux
using Flux: @functor
using Parameters
using ModelUtils

abstract type ModuleSpec end


include("./activations.jl")
include("./layers.jl")
include("./blocks.jl")
include("./heads.jl")

include("./efficientnet.jl")
include("./mobilenetv3.jl")
include("./xresnet.jl")


export
    SqueezeExcitation,

    efficientnetb0,
    mobilenetv3_small,
    mobilenetv3_large,
    xresnet18,
    xresnet50

end # module
