using BeamTracking
using Documenter

DocMeta.setdocmeta!(BeamTracking, :DocTestSetup, :(using BeamTracking); recursive=true)

makedocs(;
    modules=[BeamTracking],
    authors="mattsignorelli <mgs255@cornell.edu> and contributors",
    sitename="BeamTracking.jl",
    format=Documenter.HTML(;
        canonical="https://bmad-sim.github.io/BeamTracking.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/bmad-sim/BeamTracking.jl",
    devbranch="main",
)
