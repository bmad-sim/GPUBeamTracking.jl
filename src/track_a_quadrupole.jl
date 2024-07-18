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

x = CUDA.fill(1.0, 10_000); px = CUDA.fill(0.8, 10_000); y = CUDA.fill(1.0, 10_000);
py = CUDA.fill(0.85, 10_000); z = CUDA.fill(0.5, 10_000); pz = CUDA.fill(0.2, 10_000);
s = 1.0; p0c = CUDA.fill(1.184, 10_000); mc2 = 0.511;

NUM_STEPS = Int32(1000); K1 = 1.0; L = 0.5;
TILT = CUDA.fill(1.0, 10_000); X_OFFSET = CUDA.fill(0.006, 10_000); Y_OFFSET = CUDA.fill(0.01, 10_000);

x_ele_int = CUDA.fill(0.0, 10_000); x_ele = CUDA.fill(0.0, 10_000); y_ele = CUDA.fill(0.0, 10_000);
px_ele = CUDA.fill(0.0, 10_000); py_ele = CUDA.fill(0.0, 10_000); S = CUDA.fill(0.0, 10_000); C = CUDA.fill(0.0, 10_000);

p_in = particle(x, px, y, px, z, pz, s, p0c, mc2);
quad = quad_input(L, K1, NUM_STEPS, X_OFFSET, Y_OFFSET, TILT);
int = int_set(x_ele_int, x_ele, y_ele, px_ele, py_ele, S, C);

kernel = @cuda launch=false track_a_quadrupole(p_in, quad)


