
"""
$(TYPEDEF)

Basic convolutional module with batch normalization.

## Fields

$(TYPEDFIELDS)

## Notes

In [MobileNetV3], the activations are `relu` and `hardσ`, while in EfficientNet,
they are `swish` and `σ`.
"""
@with_kw struct ConvBlock <: ModuleSpec
    "kernel size"
    ksize::Int
    "number of input kernels"
    k_in::Int
    "number of output kernels"
    k_out::Int
    "non-linearity/activation function (applied after BN)"
    σ = relu
    "stride of the convolution"
    stride::Int = 1
end
function (cb::ConvBlock)()
    @unpack ksize, k_in, k_out, σ, stride = cb
    return Chain(
        Conv((ksize, ksize), k_in => k_out; pad = ksize ÷ 2, stride = stride),
        BatchNorm(k_out, σ),
    )

end



"""
$(TYPEDEF)

A squeeze-and-excitation block.

## Fields

$(TYPEDFIELDS)

## Notes

In [MobileNetV3], the activations are `relu` and `hardσ`, while in EfficientNet,
they are `swish` and `σ`.
"""
@with_kw struct SqueezeExcitation <: ModuleSpec
    "Number of input kernels and output kernels"
    k::Integer
    "Ratio to calculate `k_mid`"
    ratio::Integer = 4
    "Ratio to calculate `k_mid`"
    k_mid::Integer = k ÷ ratio
    "Activation function to apply after \"squeezing\""
    σ1 = relu
    "Activation function to apply after \"excitation\""
    σ2 = relu
end
function (se::SqueezeExcitation)()
    @unpack k, k_mid, σ1, σ2 = se
    return SkipConnection(
        Chain(
            GlobalMeanPool(),
            Conv((1, 1), k => k_mid),
            BatchNorm(k_mid, σ1),
            Conv((1, 1), k_mid => k),
            BatchNorm(k, σ2),
        ),
        bmul,
    )
end


"""
$(TYPEDEF)

A depthwise separable convolution block.

## Fields

$(TYPEDFIELDS)
"""
@with_kw struct DepthwiseSeparable <: ModuleSpec
    "Kernel size of convolution"
    ksize::Integer
    "Number of input kernels"
    k_in::Integer
    "Number of expanded kernels"
    k_out::Integer = 6k_in
    "Activation function to apply after convolution"
    σ = relu
    "Stride of convolution"
    stride::Integer = 1
end

function (dws::DepthwiseSeparable)()
    @unpack ksize, k_in, k_out, stride = dws
    return Chain(
        Conv((1, 1), k_in => k_out),
        BatchNorm(k_out, σ),

        DepthwiseConv(
            (ksize, ksize),
            k_out => k_out;
            pad = ksize ÷ 2,
            stride = stride
        ),
        BatchNorm(k_out, σ)
    )
end


"""
$(TYPEDEF)

Mobile Inverted Bottleneck block, as described in
[Searching for MobileNetV3](https://arxiv.org/abs/1905.02244).

Also used in [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946v3)

## Fields

$(TYPEDFIELDS)
"""
@with_kw struct MBConv <: ModuleSpec
    "Kernel size of expansion block"
    ksize::Int
    "Number of input kernels"
    k_in::Int
    "Number of kernels after expansion"
    k_exp::Int
    "Number of output kernels"
    k_out::Int
    "Non-linearity/activation function to use in expansion block"
    σ = relu
    "Stride of the expansion block"
    stride = 1
    "Whether to add a squeeze-excitation block between expansion and projection"
    has_se = true
    """
    Convolution block to use, either [`DepthwiseSeparable`](@ref) or
    [`ConvBlock`](@ref).
    Depthwise separable convolutions are used in the literature, but as of
    `Flux@0.10.4`, the `DepthwiseConv` layer does not run on GPU, see
    [Flux.jl Issue #459](https://github.com/FluxML/Flux.jl/issues/459).
    """
    convblock = ConvBlock
end
function (mbc::MBConv)()
    @unpack ksize, k_in, k_exp, k_out, σ, stride, has_se, convblock = mbc

    block = Chain(
        # expansion
        convblock(ksize = ksize, k_in = k_in, k_out = k_exp; σ = σ, stride = stride)(),
        has_se ? SqueezeExcitation(k = k_exp)() : identity,

        # projection
        Conv((1, 1), k_exp => k_out),
    )

    # add residual connection if input and output are the same size
    # as done in MobileNetV3
    if k_in == k_out && stride == 1
        block = Residual(block)
    end

    return block

end


# Utilities


"""
    bmul(x, weights) = x .* weights

Broadcasted multiplication.
"""
bmul(x, weights) = x .* weights
