"""Structures used for particle tracking on GPU;
adapt type with Adapt.@adapt_structure in main program"""

struct GPU_Particle{K}
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
    """Intermediate helper struct"""
    P::K
    Px::K
    Py::K
    Pxy2::K
    Pl::K
    dz::K
end

struct GPU_Drift{K}
    L::K
end

