

@with_kw struct MobileNetV3 <: ModuleSpec
    stages::AbstractVector{MBConv}
    "Size of last convolution"
    k_out::Int
end

function (mbnv3::MobileNetV3)()
    head = ConvBlock(ksize = 3, k_in = 3, k_out = 16; σ = hswish, stride = 2)()

    return Chain(
        head,
        [stage() for stage in mbnv3.stages]...,
        Conv((1, 1), mbnv3.stages[end].k_out => mbnv3.k_out),
        BatchNorm(mbnv3.k_out, hswish)
    )

end


"""
($TYPEDEF)

Classification head for MobileNetV3 as described in
MobileNetV3-small from [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)
section 5.1 and figure 5.

$(TYPEDFIELDS)
"""
@with_kw struct MobileNetV3Head <: ModuleSpec
    "Number of classes to predict"
    n_classes
    "Number of input kernels"
    k_in
    "Number of intermediate kernels"
    k_mid
end

function (head::MobileNetV3Head)()
    return Chain(
        GlobalMeanPool(),
        Conv((1, 1), head.k_in => head.k_mid, hswish),
        Conv((1, 1), head.k_mid => head.n_classes),
        flatten
    )
end




"""
    mobilenetv3_small(usedepthwise = false)

MobileNetV3-small from [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)
"""
function mobilenetv3_small(usedepthwise = false)
    cb = usedepthwise ? DepthwiseSeparable : ConvBlock
    mbvn3 = MobileNetV3([
        MBConv(ksize = 3, k_in = 16, k_exp = 16, k_out = 16,
            σ = relu, has_se = true, stride = 2, convblock = cb),
        MBConv(ksize = 3, k_in = 16, k_exp = 72, k_out = 24,
            σ = relu, has_se = false, stride = 2, convblock = cb),
        MBConv(ksize = 3, k_in = 24, k_exp = 88, k_out = 24,
            σ = relu, has_se = false, stride = 1, convblock = cb),
        MBConv(ksize = 5, k_in = 24, k_exp = 96, k_out = 40,
            σ = hswish, has_se = true, stride = 2, convblock = cb),
        MBConv(ksize = 5, k_in = 40, k_exp = 240, k_out = 40,
            σ = hswish, has_se = true, stride = 1, convblock = cb),
        MBConv(ksize = 5, k_in = 40, k_exp = 240, k_out = 40,
            σ = hswish, has_se = true, stride = 1, convblock = cb),
        MBConv(ksize = 5, k_in = 40, k_exp = 120, k_out = 48,
            σ = hswish, has_se = true, stride = 1, convblock = cb),
        MBConv(ksize = 5, k_in = 48, k_exp = 144, k_out = 48,
            σ = hswish, has_se = true, stride = 1, convblock = cb),
        MBConv(ksize = 5, k_in = 48, k_exp = 288, k_out = 96,
            σ = hswish, has_se = true, stride = 2, convblock = cb),
        MBConv(ksize = 5, k_in = 96, k_exp = 576, k_out = 96,
            σ = hswish, has_se = true, stride = 1, convblock = cb),
        MBConv(ksize = 5, k_in = 96, k_exp = 576, k_out = 96,
            σ = hswish, has_se = true, stride = 1, convblock = cb),
    ], 576)

    return mbvn3()
end


"""
    mobilenetv3_large(usedepthwise = false)

MobileNetV3-large from [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)
"""
function mobilenetv3_large(usedepthwise = false)
    cb = usedepthwise ? DepthwiseSeparable : ConvBlock
    mbvn3 = MobileNetV3([
        MBConv(ksize = 3, k_in = 16, k_exp = 16, k_out = 16,
            σ = relu, has_se = false, stride = 1, convblock = cb),

        MBConv(ksize = 3, k_in = 16, k_exp = 64, k_out = 24,
            σ = relu, has_se = false, stride = 2, convblock = cb),
        MBConv(ksize = 3, k_in = 24, k_exp = 72, k_out = 24,
            σ = relu, has_se = false, stride = 1, convblock = cb),

        MBConv(ksize = 5, k_in = 24, k_exp = 72, k_out = 40,
            σ = relu, has_se = true, stride = 2, convblock = cb),
        MBConv(ksize = 5, k_in = 40, k_exp = 120, k_out = 40,
            σ = relu, has_se = true, stride = 1, convblock = cb),
        MBConv(ksize = 5, k_in = 40, k_exp = 120, k_out = 40,
            σ = relu, has_se = true, stride = 1, convblock = cb),

        MBConv(ksize = 3, k_in = 40, k_exp = 240, k_out = 80,
            σ = hswish, has_se = false, stride = 2, convblock = cb),
        MBConv(ksize = 3, k_in = 80, k_exp = 200, k_out = 80,
            σ = hswish, has_se = false, stride = 1, convblock = cb),
        MBConv(ksize = 3, k_in = 80, k_exp = 184, k_out = 80,
            σ = hswish, has_se = false, stride = 1, convblock = cb),
        MBConv(ksize = 3, k_in = 80, k_exp = 184, k_out = 80,
            σ = hswish, has_se = false, stride = 1, convblock = cb),
        MBConv(ksize = 3, k_in = 80, k_exp = 480, k_out = 112,
            σ = hswish, has_se = true, stride = 1, convblock = cb),
        MBConv(ksize = 3, k_in = 112, k_exp = 672, k_out = 112,
            σ = hswish, has_se = true, stride = 1, convblock = cb),

        MBConv(ksize = 5, k_in = 112, k_exp = 672, k_out = 160,
            σ = hswish, has_se = true, stride = 2, convblock = cb),
        MBConv(ksize = 5, k_in = 160, k_exp = 960, k_out = 160,
            σ = hswish, has_se = true, stride = 1, convblock = cb),
        MBConv(ksize = 5, k_in = 160, k_exp = 960, k_out = 160,
            σ = hswish, has_se = true, stride = 1, convblock = cb),
    ], 960)

    return mbvn3()
end


function mobilenetv3_head_small(n_classes)
    return MobileNetV3Head(
        n_classes = n_classes,
        k_in = 576,
        k_mid = 1024
    )()
end


function mobilenetv3_head_large(n_classes)
    return MobileNetV3Head(
        n_classes = n_classes,
        k_in = 576,
        k_mid = 1024
    )()
end
