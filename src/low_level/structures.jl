using CUDA
"""Structures used for particle tracking"""

struct GPU_Particle{K}
    """Particle struct on GPU; adapt type
    with Adapt.@adapt_structure in main program"""
    x::K
    px::K
    y::K
    py::K
    z::K
    pz::K
    s::K
    p0c::K
    mc2::K

end

struct Intermediate{K}
    P::K
    Px::K
    Py::K
    Pxy2::K
    Pl::K
    dz::K
end

struct GPU_Drift{K}
    """Drift structure on GPU"""
    L::K
end

struct Drift{T}
    L::T
end

struct Particle{T}
    x::T
    px::T
    y::T
    py::T
    z::T
    pz::T
    s::T
    p0c::T
    mc2::T
end