 """Structures used for particle tracking on GPU;
adapt type with Adapt.@adapt_structure in main program"""

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

struct Drift{T}
    L::T
end

struct offset_and_tilt{T}
    x_offset::T
    y_offset::T
    tilt::T
end

struct z_correction{T}
    pz::T
    p0c::T
    mass::T
    ds::T
end

"""Intermediate structs in order 
to not dynamically allocate memory to
intermediate calculations"""

struct Intermediate_Drift{T}
    P::T
    Px::T
    Py::T
    Pxy2::T
    Pl::T
    dz::T
end

struct Intermediate_Set{T}
    x_ele_int::T
    x_ele::T
    y_ele::T
    px_ele::T
    py_ele::T
    s::T
    c::T
end

struct Intermediate_Unset{T}
    x_lab::T
    y_lab::T
    px_lab::T
    py_lab::T
    s::T
    c::T
end

struct Intermediate_z_Correction{T}
    beta::T
    beta0::T
    e_tot::T
    evaluation::T
    dz::T
end