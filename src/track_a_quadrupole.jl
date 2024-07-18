using CUDA
include("low_level/structures.jl"); include("low_level/offset_particle.jl"); include("low_level/int_arrays.jl");

fill_quadrupole(10_000);

function track_a_quadrupole(p_in, quad)
    """Tracks the incoming Particle p_in though quad element and
    returns the outgoing particle.
    See Bmad manual section 24.15
    """
    l = quad.L
    k1 = quad.K1
    n_step = quad.NUM_STEPS  # number of divisions
    step_len = l / n_step  # length of division

    x_off = quad.X_OFFSET
    y_off = quad.Y_OFFSET
    tilt = quad.TILT

    corrections = offset_and_tilt(x_off, y_off, tilt)

    s = p_in.s
    p0c = p_in.p0c
    mc2 = p_in.mc2
    
    # --- TRACKING --- :
    @cuda threads=768 blocks=14 dynamic=true offset_particle_set(corrections, p_in, int)
    x, px, y, py, z, pz = x_ele, px_ele, y_ele, py_ele, p_in.z, p_in.pz

    return nothing
end



