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

struct Intermediate_Drift{K}
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

struct Intermediate_Offset{K}
    x_ele_int::K
    y_ele_int::K
    x_ele::K
    y_ele::K
    px_ele::K
    py_ele::K
    s::K
    c::K
end

struct offset_and_tilt{K}
    x_offset::K
    y_offset::K
    tilt::K
end




