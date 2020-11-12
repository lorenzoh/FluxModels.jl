
"""
    AdaptiveMeanPool(target_out)
Adaptive mean pooling layer. 'target_out' stands for the size of one layer(channel) of output.
Irrespective of Input size and pooling window size, it returns the target output dimensions.
"""
struct AdaptiveMeanPool{N}
  target_out::NTuple{N, Int}
end

function (m::AdaptiveMeanPool)(x)
  w = size(x, 1) - m.target_out[1] + 1
  h = size(x, 2) - m.target_out[2] + 1
  pdims = PoolDims(x, (w, h); padding = (0, 0), stride = (1, 1))
  return meanpool(x, pdims)
end

"""
    AdaptiveMaxPool(target_out)
Adaptive max pooling layer. 'target_out' stands for the size of one layer(channel) of output.
Irrespective of Input size and pooling window size, it returns the target output dimensions.
"""
struct AdaptiveMaxPool{N}
  target_out::NTuple{N, Int}
end

function (m::AdaptiveMaxPool)(x)
  w = size(x, 1) - m.target_out[1] + 1
  h = size(x, 2) - m.target_out[2] + 1
  pdims = PoolDims(x, (w, h); padding = (0, 0), stride = (1, 1))
  return maxpool(x, pdims)
end

"""
  flatten(x)

Flatten `x` in all dimensions except the last/batch dimension
"""
flatten(x) = reshape(x, :, size(x)[end])



"""
    Residual(model) == SkipConnection(model, +)

Helper for defining residual connections
"""
Residual(model) = SkipConnection(model, +)

function pixelshuffle(x::AbstractArray{T, 4}, ratios = (2, 2)) where T
    iH, iW, C, B = size(x)
    pH, pW = ratios
    oC, oH, oW = C ÷ (pH*pW), iH * pH, iW * pW

    x = reshape(x, iH, iW, pH, pW, oC, B)
    x = permutedims(x, (4, 1, 3, 2, 5, 6))  # pH, iH, pW, iW, oC, B
    x = reshape(x, oH, oW, oC, B)

    return x
end

"""
    PixelShuffle(scale, kernels_in[, kernels_out])

Pixel shuffle layer that upscales height and width of `x` by `scale`. Has reduced
checkerboard artifacts compared to `ConvTranspose`

Introduced in [Real-Time Single Image and Video Super-Resolution Using
an EfficientSub-Pixel Convolutional Neural Network](https://arxiv.org/pdf/1609.05158.pdf).

"""
struct PixelShuffle
    conv::Conv
    scales::Tuple{Int, Int}
end

Flux.@functor PixelShuffle

function PixelShuffle(scales::Tuple{Int, Int}, k_in, k_out = k_in)
    return PixelShuffle(
        Conv((1, 1), k_in => k_out * scales[1] * scales[2]),
        scales,
    )
end

PixelShuffle(scale::Int, k_in, k_out = k_in) = PixelShuffle((scale, scale), k_in, k_out)

function (ps::PixelShuffle)(x)
    pixelshuffle(ps.conv(x), ps.scales)
end
