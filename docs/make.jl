using FluxModels
using Documenter

makedocs(;
    modules=[FluxModels],
    authors="lorenzoh <lorenz.ohly@gmail.com>",
    repo="https://github.com/lorenzoh/FluxModels.jl/blob/{commit}{path}#L{line}",
    sitename="FluxModels.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://lorenzoh.github.io/FluxModels.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/lorenzoh/FluxModels.jl",
)
