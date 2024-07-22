using CUDA

include("low_level/structures.jl"); include("low_level/offset_particle.jl"); include("low_level/int_arrays.jl");

function track_a_quadrupole!(p_in, quad, int)
    """Tracks the incoming Particle p_in though quad element and
    returns the outgoing particle.
    See Bmad manual section 24.15
    """
    l = quad.L
    x_off = quad.X_OFFSET
    y_off = quad.Y_OFFSET
    tilt = quad.TILT
    k1 = quad.K1
    n_step = quad.NUM_STEPS  # number of divisions
    l /= n_step  # length of division

    x_ele, px_ele, S, C, sqrt_k, sk_l, sx, ax11, ax12, ax21, cx1, 
    cx2, cx3, ay11, ay12, ay21, cy1, cy2, cy3, b1, rel_p = int.x_ele, 
    int.px_ele, int.S, int.C, int.sqrt_k, int.sk_l, int.sx, int.ax11, 
    int.ax12, int.ax21, int.cx1, int.cx2, int.cx3, int.ay11, int.ay12, 
    int.ay21, int.cy1, int.cy2, int.cy3, int.b1, int.rel_p
    
    # --- TRACKING --- :
    x, px, y, py, z, pz = p_in.x, p_in.px, p_in.y, p_in.py, p_in.z, p_in.pz

    eps = 2.220446049250313e-16  # machine epsilon to double precision
    
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    """offset_particle_set"""
    i = index
    while i <= length(x)
        
        # set to particle coordinates
        @inbounds (b1[i] = k1[i] * l;
        
        S[i] = sin(tilt[i]);
        C[i] = cos(tilt[i]);
        x[i] -= x_off[i];
        y[i] -= y_off[i];
        x_ele[i] = x[i]*C[i] + y[i]*S[i]; 
        y[i] = -x[i]*S[i] + y[i]*C[i];
        px_ele[i] = px[i]*C[i] + py[i]*S[i];
        py[i] *= C[i];
        py[i] -= px[i]*S[i];

        # transfer matrix elements and coefficients for x-coord
        rel_p[i] = 1 + pz[i];
        k1[i] = b1[i]/(l*rel_p[i]);
        
        sqrt_k[i] = sqrt(abs(-k1[i])+eps);
        sk_l[i] = sqrt_k[i] * l;
        
        ax11[i] = cos(sk_l[i]) * (-k1[i]<=0) + cosh(sk_l[i]) * (-k1[i]>0);
        sx[i] = (sin(sk_l[i])/(sqrt_k[i]))*(-k1[i]<=0) + (sinh(sk_l[i])/(sqrt_k[i]))*(-k1[i]>0);
            
        ax12[i] = sx[i] / rel_p[i];
        ax21[i] = -k1[i] * sx[i] * rel_p[i];
            
        cx1[i] = -k1[i] * (-ax11[i] * sx[i] + l) / 4;
        cx2[i] = k1[i] * sx[i]^2 / (2 * rel_p[i]);
        cx3[i] = -(ax11[i] * sx[i] + l) / (4 * rel_p[i]^2);
        
        # transfer matrix elements and coefficients for y-coord
        ay11[i] = cos(sk_l[i]) * (k1[i]<=0) + cosh(sk_l[i]) * (k1[i]>0);
        sx[i] = (sin(sk_l[i])/(sqrt_k[i]))*(k1[i]<=0) + (sinh(sk_l[i])/(sqrt_k[i]))*(k1[i]>0);
            
        ay12[i] = sx[i] / rel_p[i];
        ay21[i] = k1[i] * sx[i] * rel_p[i];
            
        cy1[i] = k1[i] * (-ay11[i] * sx[i] + l) / 4;
        cy2[i] = -k1[i] * sx[i]^2 / (2 * rel_p[i]);
        cy3[i] = -(ay11[i] * sx[i] + l) / (4 * rel_p[i]^2);)

        i += stride 

    end

    
        
    

    """continue"""
    s = p_in.s
    p0c = p_in.p0c
    mc2 = p_in.mc2
    
    return nothing
end

fill_quadrupole(10_000);
x = CUDA.fill(1.0, 10_000); px = CUDA.fill(0.8, 10_000); y = CUDA.fill(1.0, 10_000);
py = CUDA.fill(0.85, 10_000); z = CUDA.fill(0.5, 10_000); pz = CUDA.fill(0.2, 10_000);
s = 1.0; p0c = CUDA.fill(1.184, 10_000); mc2 = 0.511;

NUM_STEPS = Int32(1000); K1 = CUDA.fill(1.0, 10_000); L = 0.5;
TILT = CUDA.fill(1.0, 10_000); X_OFFSET = CUDA.fill(0.006, 10_000); Y_OFFSET = CUDA.fill(0.01, 10_000); 


p_in = particle(x, px, y, py, z, pz, s, p0c, mc2);
quad = quad_input(L, K1, NUM_STEPS, X_OFFSET, Y_OFFSET, TILT);
int = int_quad_elements(x_ele, px_ele, S, C, sqrt_k, sk_l, sx, ax11, ax12, 
                        ax21, ay11, ay12, ay21, cx1, cx2, cx3, cy1, cy2, cy3, b1, rel_p);

@cuda threads=1024 blocks=10 track_a_quadrupole!(p_in, quad, int)
print(x_ele[1:3])
print(y[1:3])
print(cx1[1:3])