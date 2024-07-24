using CUDA

include("low_level/structures.jl"); include("low_level/int_arrays.jl");

function track_a_quadrupole!(p_in, quad, int)
    """Tracks the incoming Particle p_in though quad element and
    returns the outgoing particle.
    See Bmad manual section 24.15
    """
    len = quad.L
    x_off = quad.X_OFFSET
    y_off = quad.Y_OFFSET
    tilt = quad.TILT
    K1 = quad.K
    n_step = quad.NUM_STEPS  # number of divisions
    l = len/n_step # length of division

    s = p_in.s
    p0c = p_in.p0c
    mc2 = p_in.mc2

    x_ele, px_ele, S, C, sqrt_k, sk_l, sx, a11, a12, a21, c1, 
    c2, c3, b1, rel_p = int.x_ele, int.px_ele, int.S, int.C, int.sqrt_k, int.sk_l, 
    int.sx, int.a11, int.a12, int.a21, int.c1, int.c2, int.c3, int.b1, int.rel_p
    
    # --- TRACKING --- :
    x, px, y, py, z, pz = p_in.x, p_in.px, p_in.y, p_in.py, p_in.z, p_in.pz

    eps = 2.220446049250313e-16  # machine epsilon to double precision
    
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    """offset_particle_set"""
    i = index
    j = Int32(1)
    while i <= length(x)
        
        # set to particle coordinates
        @inbounds (
        b1[i] = K1[i] * len;
        S[i] = sin(tilt[i]);
        C[i] = cos(tilt[i]);
        x[i] -= x_off[i];
        y[i] -= y_off[i];
        x_ele[i] = x[i]*C[i] + y[i]*S[i]; 
        y[i] = -x[i]*S[i] + y[i]*C[i];
        px_ele[i] = px[i]*C[i] + py[i]*S[i];
        py[i] *= C[i];
        py[i] -= px[i]*S[i];)

        while j <= n_step
            # transfer matrix elements and coefficients for x-coord
            @inbounds (
            rel_p[i] = 1.0 + pz[i];
            K1[i] = b1[i]/(rel_p[i]*len);

            sqrt_k[i] = sqrt(abs(K1[i])+eps);
            sk_l[i] = sqrt_k[i] * l;
            
            K1[i] *= -1;

            a11[i] = cos(sk_l[i]) * (K1[i]<=0) + cosh(sk_l[i]) * (K1[i]>0);
            sx[i] = (sin(sk_l[i])/(sqrt_k[i]))*(K1[i]<=0) + (sinh(sk_l[i])/(sqrt_k[i]))*(K1[i]>0);

            a12[i] = sx[i] / rel_p[i];
            a21[i] = K1[i] * sx[i] * rel_p[i];
                
            c1[i] = K1[i] * (-a11[i] * sx[i] + l) / 4;
            c2[i] = -K1[i] * sx[i]^2 / (2 * rel_p[i]);
            c3[i] = -(a11[i] * sx[i] + l) / (4 * rel_p[i]^2);

            # z (without energy correction)
            z[i] += (c1[i] * x_ele[i]^2 + c2[i] * x_ele[i] * px_ele[i] + c3[i] 
            * px_ele[i]^2);
            
            # next index x-vals
            x_ele[i] = a11[i] * x_ele[i] + a12[i] * px_ele[i];
            px_ele[i] = a21[i] * x_ele[i] + a11[i] * px_ele[i];
            
            K1[i] *= -1;
            # transfer matrix elements and coefficients for y-coord
            a11[i] = cos(sk_l[i]) * (K1[i]<=0) + cosh(sk_l[i]) * (K1[i]>0);
            sx[i] = (sin(sk_l[i])/(sqrt_k[i]))*(K1[i]<=0) + (sinh(sk_l[i])/(sqrt_k[i]))*(K1[i]>0);
            
            a12[i] = sx[i] / rel_p[i];
            a21[i] = K1[i] * sx[i] * rel_p[i];
                
            c1[i] = K1[i] * (-a11[i] * sx[i] + l) / 4;
            c2[i] = -K1[i] * sx[i]^2 / (2 * rel_p[i]);
            c3[i] = -(a11[i] * sx[i] + l) / (4 * rel_p[i]^2);
            
            # z (without energy correction)
            z[i] += c1[i] * y[i]^2 + c2[i] * y[i] * py[i] + c3[i] * py[i]^2;
            
            # next index vals
            y[i] = a11[i] * y[i] + a12[i] * py[i];
            py[i] = a21[i] * y[i] + a11[i] * py[i];)
            j += 1
            
        end
        
        i += stride

    end
    
    
    return nothing
end

fill_quadrupole(10_000);
x = CUDA.fill(1.0, 10_000); px = CUDA.fill(0.8, 10_000); y = CUDA.fill(1.0, 10_000);
py = CUDA.fill(0.85, 10_000); z = CUDA.fill(0.5, 10_000); pz = CUDA.fill(0.2, 10_000);
s = 1.0; p0c = CUDA.fill(1.184, 10_000); mc2 = 0.511;

NUM_STEPS = Int32(1000); K1 = CUDA.fill(2.0, 10_000); L = 0.2;
TILT = CUDA.fill(1.0, 10_000); X_OFFSET = CUDA.fill(0.006, 10_000); Y_OFFSET = CUDA.fill(0.01, 10_000);

p_in = particle(x, px, y, py, z, pz, s, p0c, mc2);
quad = quad_and_sextupole(L, K1, NUM_STEPS, X_OFFSET, Y_OFFSET, TILT);
int = int_quad(x_ele, px_ele, S, C, sqrt_k, sk_l, sx, a11, a12,
                        a21, c1, c2, c3, b1, rel_p);

kernel = @cuda launch=false track_a_quadrupole!(p_in, quad, int);
config = launch_configuration(kernel.fun)
threads = config.threads
blocks= cld(length(x), threads)

@cuda threads=threads blocks=blocks track_a_quadrupole!(p_in, quad, int)

print(c1[1:3])
print(c2[1:3])
print(c3[1:3])
print(sqrt_k[1:3])
print(sk_l[1:3])
print(a11[1:3])
print(sx[1:2])
print(a12[1:2])
print(K1[1:3])
print(b1[1:3])
print(rel_p[1:3])
