
module OutputSizes

using Flux
using LinearAlgebra


"""
    Nil <: Number

Nil is a singleton type with a single instance `nil`. Unlike
`Nothing` and `Missing` it subtypes `Number`.

"""
struct Nil <: Number end

const nil = Nil()

Nil(::T) where T<:Number = nil

Base.copy(::Nil) = nil

Base.:+(::Nil) = nil
Base.:-(::Nil) = nil

Base.:+(::Nil, ::Nil) = nil
Base.:+(::Nil, ::Number) = nil
Base.:+(::Number, ::Nil) = nil

Base.:-(::Nil, ::Nil) = nil
Base.:-(::Nil, ::Number) = nil
Base.:-(::Number, ::Nil) = nil

Base.:*(::Nil, ::Nil) = nil
Base.:*(::Nil, ::Number) = nil
Base.:*(::Number, ::Nil) = nil

Base.:/(::Nil, ::Nil) = nil
Base.:/(::Nil, ::Number) = nil
Base.:/(::Number, ::Nil) = nil

Base.inv(::Nil) = nil

Base.isless(::Nil, ::Nil) = true
Base.isless(::Nil, ::Number) = true
Base.isless(::Number, ::Nil) = true

Base.abs(::Nil) = nil
Base.exp(::Nil) = nil

Base.typemin(::Type{Nil}) = nil
Base.typemax(::Type{Nil}) = nil
Base.:^(::Nil, ::Nil) = nil

Base.promote(x::Nil, y::Nil) = (nil, nil)
Base.promote(x::Nil, y) = (nil, nil)
Base.promote(x, y::Nil) = (nil, nil)
Base.promote(x::Nil, y, z) = (nil, nil, nil)
Base.promote(x, y::Nil, z) = (nil, nil, nil)
Base.promote(x, y, z::Nil) = (nil, nil, nil)
Base.promote(x::Nil, y, z::Nil) = (nil, nil, nil)
Base.promote(x::Nil, y::Nil, z::Nil) = (nil, nil, nil)
Base.promote(x::Nil, y::Nil, z) = (nil, nil, nil)


LinearAlgebra.adjoint(::Nil) = nil
LinearAlgebra.transpose(::Nil) = nil

# Since some Flux layers don't work out of the box,
# we're "cheating" here by implementing the layers
# for `Nil` with the correct output size.


function (c::Flux.Conv)(x::AbstractArray{Nil})
    return fill(nil, Flux.outdims(c, size(x))..., size(c.weight, 4), size(x, 4))
end

function (c::Flux.DepthwiseConv)(x::AbstractArray{Nil})
    return fill(nil, Flux.outdims(c, size(x))..., size(c.weight, 4), size(x, 4))
end

function (c::Flux.ConvTranspose)(x::AbstractArray{Nil})
    return fill(nil, Flux.outdims(c, size(x))..., size(c.weight, 4), size(x, 4))
end


# Interface

function outputsize(model, sz; addbatch = true)
    return obssize(model(fill(nil, sz..., 1)))
end

# Utils

obssize(a::AbstractArray) = size(a)[1:end-1]
obssize(t::Union{Tuple, NamedTuple}) = map(obssize, t)

export nil, Nil, outputsize

end  # module
