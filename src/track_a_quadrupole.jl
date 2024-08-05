using CUDA

include("low_level/structures.jl"); include("low_level/int_arrays.jl");

function track_a_quadrupole!(p_in, quad, int)
    """Tracks the incoming Particle p_in though quad element and
    returns the outgoing particle.
    See Bmad manual section 24.15
    """
    l = quad.L
    x_off = quad.X_OFFSET
    y_off = quad.Y_OFFSET
    tilt = quad.TILT
    K1 = quad.K
    n_step = quad.NUM_STEPS  # number of divisions
    step_len = l/n_step # length of division

    s = p_in.s
    p0c = p_in.p0c
    mc2 = p_in.mc2

    x_ele, px_ele, y_ele, S, C, sqrt_k, sk_l, sx, a11, a12, a21, c1, 
    c2, c3, rel_p, beta, beta0, e_tot, evaluation, dz = int.x_ele, 
    int.px_ele, int.y_ele, int.S, int.C, int.sqrt_k, int.sk_l, int.sx, int.a11, 
    int.a12, int.a21, int.c1, int.c2, int.c3, int.rel_p, int.beta, int.beta0,
    int.e_tot, int.evaluation, int.dz
    
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
        K1[i] /= (1 + pz[i]);

        S[i] = sin(tilt[i]);
        C[i] = cos(tilt[i]);
        x[i] -= x_off[i];
        y[i] -= y_off[i];
        x_ele[i] = x[i]*C[i] + y[i]*S[i]; 
        y[i] = -x[i]*S[i] + y[i]*C[i];
        px_ele[i] = px[i]*C[i] + py[i]*S[i];
        py[i] *= C[i];
        py[i] -= px[i]*S[i];
        
        x[i] = x_ele[i];
        px[i] = px_ele[i];)

        while j <= n_step
            # transfer matrix elements and coefficients for x-coord
            @inbounds (
            K1[i] *= -1;
            
            sqrt_k[i] = sqrt(abs(K1[i]+eps));
            sk_l[i] = sqrt_k[i] * step_len;
            

            a11[i] = cos(sk_l[i]) * (K1[i]<=0) + cosh(sk_l[i]) * (K1[i]>0);
            sx[i] = (sin(sk_l[i])/(sqrt_k[i]))*(K1[i]<=0) + (sinh(sk_l[i])/(sqrt_k[i]))*(K1[i]>0);

            a12[i] = sx[i] / rel_p[i];
            a21[i] = K1[i] * sx[i] * rel_p[i];
                
            c1[i] = K1[i] * (-a11[i] * sx[i] + step_len) / 4;
            c2[i] = -K1[i] * sx[i]^2 / (2 * rel_p[i]);
            c3[i] = -(a11[i] * sx[i] + step_len) / (4 * rel_p[i]^2);

            # z (without energy correction)
            z[i] += (c1[i] * x[i]^2 + c2[i] * x[i] * px[i] + c3[i] 
            * px[i]^2);
            
            # next index x-vals
            x_ele[i] = a11[i] * x[i] + a12[i] * px[i];
            px_ele[i] = a21[i] * x[i] + a11[i] * px[i];
            
            
            K1[i] *= -1;

            sqrt_k[i] = sqrt(abs(K1[i]+eps));
            sk_l[i] = sqrt_k[i] * step_len;
            
            # transfer matrix elements and coefficients for y-coord
            a11[i] = cos(sk_l[i]) * (K1[i]<=0) + cosh(sk_l[i]) * (K1[i]>0);
            sx[i] = (sin(sk_l[i])/(sqrt_k[i]))*(K1[i]<=0) + (sinh(sk_l[i])/(sqrt_k[i]))*(K1[i]>0);
            
            a12[i] = sx[i] / rel_p[i];
            a21[i] = K1[i] * sx[i] * rel_p[i];
                
            c1[i] = K1[i] * (-a11[i] * sx[i] + step_len) / 4;
            c2[i] = -K1[i] * sx[i]^2 / (2 * rel_p[i]);
            c3[i] = -(a11[i] * sx[i] + step_len) / (4 * rel_p[i]^2);
            
            # z (without energy correction)
            z[i] += c1[i] * y[i]^2 + c2[i] * y[i] * py[i] + c3[i] * py[i]^2;
            
            # next index vals
            y_ele[i] = a11[i] * y[i] + a12[i] * py[i];
            py[i] = a21[i] * y[i] + a11[i] * py[i];
            
            x[i] = x_ele[i];
            px[i] = px_ele[i];
            y[i] = y_ele[i];
            
            # z low energy correction
            beta[i] = (1+pz[i]) * p0c[i] / sqrt(((1+pz[i])*p0c[i])^2 + mc2^2);
            beta0[i] = p0c[i] / sqrt( p0c[i]^2 + mc2^2);
            e_tot[i] = sqrt(p0c[i]^2+mc2^2);
            
            evaluation[i] = mc2 * (beta0[i]*pz[i])^2;
            dz[i] = (step_len * pz[i] * (1 - 3*(pz[i]*beta0[i]^2)/2+pz[i]^2*beta0[i]^2
                    * (2*beta0[i]^2-(mc2/e_tot[i])^2/2) )
                    * (mc2/e_tot[i])^2
                    * (evaluation[i]<3e-7*e_tot[i]) 
                    + (step_len*(beta[i]-beta0[i])/beta0[i])
                    * (evaluation[i]>=3e-7*e_tot[i])  );
            
            z[i] += dz[i];
           )
            j += 1
        end

        # setting back to lab frame
        @inbounds (
        x_ele[i] = x[i]*C[i] - y[i]*S[i];
        y[i] = x[i]*S[i] + y[i]*C[i];
        x[i] = x_ele[i] + x_off[i];
        y[i] = y[i] + y_off[i];
        px_ele[i] = px[i]*C[i] - py[i]*S[i];
        py[i] = px[i]*S[i] + py[i]*C[i];
        px[i] = px_ele[i];  )

        i += stride
    end
    
    s = p_in.s + l
    return nothing
end