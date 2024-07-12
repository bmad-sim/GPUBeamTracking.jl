using Adapt

struct particle{T} # used in several scripts
    x::T
    px::T
    y::T
    py::T
    z::T
    pz::T
    sec::Float64
    p0c::T
    mc2::Float64
end

struct drift{T} # track_a_drift
    L::T
end

struct offset_and_tilt{T} # offset_particle
    x_offset::T
    y_offset::T
    tilt::T
end

struct z_correction{T} # z energy correction
    pz::T
    p0c::T
    mass::T
    ds::T
end

struct quad_calc_input{T}  # quad_mat2_calc
    k1::T
    len::Float64
    rel_p::T
end

struct quad_input{T}  #track_a_quadrupole
    L::Float64
    K1::T
    NUM_STEPS::Number
    X_OFFSET::T
    Y_OFFSET::T
    TILT::T
end


"""Intermediate structs in order 
to not dynamically allocate memory to
intermediate calculations"""

struct int_drift{T} # track_a_drift
    P::T
    Px::T
    Py::T
    Pxy2::T
    Pl::T
    dz::T
end

struct int_set{T} # offset_particle
    x_ele_int::T
    x_ele::T
    y_ele::T
    px_ele::T
    py_ele::T
    s::T
    c::T
end

struct int_unset{T}
    x_lab::T
    y_lab::T
    px_lab::T
    py_lab::T
end

struct int_z_correction{T} # z energy correction
    beta::T
    beta0::T
    e_tot::T
    evaluation::T
    dz::T
end

struct int_elements{T} # quad_mat2_calc intermediate steps
    sqrt_k::T
    sk_l::T
    cx::T
    sx::T
    a11::T
    a12::T
    a21::T
    a22::T
    c1::T
    c2::T
    c3::T
end

"""adapting structs to bitstype"""
Adapt.@adapt_structure particle; Adapt.@adapt_structure drift; Adapt.@adapt_structure offset_and_tilt;
Adapt.@adapt_structure z_correction; Adapt.@adapt_structure quad_calc_input; Adapt.@adapt_structure quad_input;
Adapt.@adapt_structure int_drift; Adapt.@adapt_structure int_set; Adapt.@adapt_structure int_unset;
Adapt.@adapt_structure int_z_correction; Adapt.@adapt_structure int_elements;

"""Structures used for particle tracking on GPU;
adapt type with Adapt.@adapt_structure in main program"""

