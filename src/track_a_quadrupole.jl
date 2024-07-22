using CUDA

include("low_level/structures.jl"); include("low_level/offset_particle.jl"); include("low_level/int_arrays.jl");

function track_a_quadrupole!(p_in, quad, int)
    """Tracks the incoming Particle p_in though quad element and
    returns the outgoing particle.
    See Bmad manual section 24.15
    """

    x_off = quad.X_OFFSET
    y_off = quad.Y_OFFSET
    tilt = quad.TILT

    x_ele, y_ele, px_ele, py_ele, S, C = int.x_ele, int.y_ele, int.px_ele, int.py_ele, int.S, int.C
    
    # --- TRACKING --- :
    x, px, y, py = p_in.x, p_in.px, p_in.y, p_in.py
    
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    """offset_particle_set"""
    i = index
    while i <= length(x)

        @inbounds (S[i] = sin(tilt[i]);
        C[i] = cos(tilt[i]);
        x[i] -= x_off[i];
        y[i] -= y_off[i];
        x_ele[i] = x[i]*C[i] + y[i]*S[i]; 
        y[i] = -x[i]*S[i] + y[i]*C[i];
        px_ele[i] = px[i]*C[i] + py[i]*S[i];
        py[i] *= C[i];
        py[i] -= px[i]*S[i];)
        i += stride
    end

    x, px, z, pz = x_ele, px_ele, p_in.z, p_in.pz

    i = index
    while i <= length(x)
        
        int = int_elements(sqr_k, sk_l, cx, sx, a11, a12, a21, a22, c1, c2, c3)
        
        sqrt_k, sk_l, cx, sx, a11, a12, a21, a22, c1, c2, c3 = int.sqrt_k, int.sk_l,
        int.cx, int.sx, int.a11, int.a12, int.a21, int.a22, int.c1, int.c2, int.c3


        for j in range(n_step)
        """quad_mat2_calc, y transfer
        matrix elements and coefficients"""
            eps = 2.220446049250313e-16  # machine epsilon to double precision
        
            sqrt_k[i] = sqrt(abs(k1)+eps)
            sk_l[i] = sqrt_k[i] * len
            
            cx[i] = cos(sk_l[i]) * (k1<=0) + cosh(sk_l[i]) * (k1>0) 
            sx[i] = (sin(sk_l[i])/(sqrt_k[i]))*(k1<=0) + (sinh(sk_l[i])/(sqrt_k[i]))*(k1>0)
                
            a11[i] = cx[i]
            a12[i] = sx[i] / rel_p[i]
            a21[i] = k1 * sx[i] * rel_p[i]
            a22[i] = cx[i]
                
            c1[i] = k1 * (-cx[i] * sx[i] + l) / 4
            c2[i] = -k1 * sx[i]^2 / (2 * rel_p[i])
            c3[i] = -(cx[i] * sx[i] + l) / (4 * rel_p[i]^2)
        end
        i += stride   
    end

    l = quad.L
    k1 = quad.K1
    n_step = quad.NUM_STEPS  # number of divisions
    l /= n_step  # length of division
    
        
    

    """continue"""
    s = p_in.s
    p0c = p_in.p0c
    mc2 = p_in.mc2
    
    return nothing
end

