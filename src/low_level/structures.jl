using Adapt

struct particle{T} # used in several scripts
    x::T
    px::T
    y::T
    py::T
    z::T
    pz::T
    s::Float64
    p0c::T
    mc2::Float64
end

struct drift{T} # track_a_drift
    L::Float64
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
    ds::Float64
end

struct quad_calc_input{T}  # quad_mat2_calc
    k1::T
    len::Float64
    rel_p::T
end

struct quad_and_sextupole{T}  #track_a_quadrupole
    L::Float64
    K::T
    NUM_STEPS::Int32
    X_OFFSET::T
    Y_OFFSET::T
    TILT::T
end

struct cavity{T}
    L::Float64
    X_OFFSET::T
    Y_OFFSET::T
    TILT::T
    VOLTAGE::T
    PHI0::T
    RF_FREQUENCY::Float64
end


"""Intermediate calculation structs
producing temporary CuArrays"""

struct int_drift{T} # track_a_drift
    P::T
    Px::T
    Py::T
    Pxy2::T
    Pl::T
    dz::T
end

struct int_set{T} # offset_particle
    x_ele::T
    px_ele::T
    S::T
    C::T
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

struct int_quad{T} 
    x_ele::T
    px_ele::T
    y_ele::T
    S::T
    C::T
    sqrt_k::T
    sk_l::T
    sx::T
    a11::T
    a12::T
    a21::T
    c1::T
    c2::T
    c3::T
    rel_p::T
    beta::T
    beta0::T
    e_tot::T
    evaluation::T
    dz::T
end

struct int_sextupole{T}
    x_ele::T
    px_ele::T
    y_ele::T
    S::T
    C::T
    beta::T
    beta0::T
    e_tot::T
    evaluation::T
    dz::T
end

struct rf_time{T}
    z::T
    pz::T
    p0c::T
    mc2::Float64
    beta::T
    time::T
end

struct energy_kick{T}
    beta::T
    E::T
    E_old::T
    pc::T
end

struct int_rf{T}
    x_ele::T
    px_ele::T
    S::T
    C::T
    phase::T
    pc::T
    beta::T
    E::T
    E_old::T
    dE::T
    time::T
    z_old::T
    dz::T
    P::T
    Px::T
    Py::T
    Pxy2::T
    Pl::T
end

struct int_crab{T}
    x_ele::T
    px_ele::T
    S::T
    C::T
    P::T
    Px::T
    Py::T
    Pxy2::T
    Pl::T
    phase::T
    beta::T
    time::T
    E::T
    pc::T
    dz::T
end


"""adapting structs to bitstype"""
Adapt.@adapt_structure particle; Adapt.@adapt_structure drift; Adapt.@adapt_structure offset_and_tilt;
Adapt.@adapt_structure z_correction; Adapt.@adapt_structure quad_calc_input; Adapt.@adapt_structure quad_and_sextupole;
Adapt.@adapt_structure int_drift; Adapt.@adapt_structure int_set; Adapt.@adapt_structure int_unset;
Adapt.@adapt_structure int_z_correction; Adapt.@adapt_structure int_quad; Adapt.@adapt_structure int_sextupole;
Adapt.@adapt_structure rf_time; Adapt.@adapt_structure energy_kick; Adapt.@adapt_structure cavity;
Adapt.@adapt_structure int_rf; Adapt.@adapt_structure int_crab;

