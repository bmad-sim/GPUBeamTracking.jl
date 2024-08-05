using CUDA

include("low_level/structures.jl"); include("low_level/int_arrays.jl");

function track_a_sextupole!(p_in, sextupole, int)
    """Tracks the incoming Particle p_in though pure sextupole element and
    returns the outgoing particle.
    See Bmad manual section 24.15
    """
    l = sextupole.L
    k2 = sextupole.K

    n_step = sextupole.NUM_STEPS  # number of divisions
    step_len = l / n_step  # length of division
    
    x_off = sextupole.X_OFFSET
    y_off = sextupole.Y_OFFSET
    tilt = sextupole.TILT
    
    p0c = p_in.p0c
    mc2 = p_in.mc2

    x_ele, px_ele, y_ele, S, C, beta, beta0, e_tot, evaluation,
    dz = int.x_ele, int.px_ele, int.y_ele, int.S, int.C,
    int.beta, int.beta0, int.e_tot, int.evaluation, int.dz
    
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    i = index
    j = Int32(1)
    # --- TRACKING --- :
    
    x, px, y, py, z, pz = p_in.x, p_in.px, p_in.y, p_in.py, p_in.z, p_in.pz
    
    while i <= length(x)

        # set to particle coordinates
        @inbounds ( 
        k2[i] /= (1 + pz[i]);  # sextupole strength divided by relative momentum (P/P0)

        S[i] = sin(tilt[i]);
        C[i] = cos(tilt[i]);
        x[i] -= x_off[i];
        y[i] -= y_off[i];
        x_ele[i] = x[i]*C[i] + y[i]*S[i]; 
        y[i] = -x[i]*S[i] + y[i]*C[i];
        px_ele[i] = px[i]*C[i] + py[i]*S[i];
        py[i] = py[i]*C[i] - px[i]*S[i];

        x[i] = x_ele[i];
        px[i] = px_ele[i]; )


        while j <= n_step
            @inbounds (
            # next index
            x_ele[i] = x[i] + step_len * px[i];
            y_ele[i] = y[i] + step_len * py[i];

            px_ele[i] = px[i] + 0.5 * k2[i] * step_len * (y[i]^2 - x[i]^2);
            py[i] = py[i] + k2[i] * step_len * x[i] * y[i];
            
            x[i] = x_ele[i];
            y[i] = y_ele[i];
            px[i] = px_ele[i];
            
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
        x_ele[i] = x[i]*S[i] - y[i]*C[i];
        y[i] = x[i]*C[i] + y[i]*S[i];
        x[i] = x_ele[i] + x_off[i];
        y[i] = y[i] + y_off[i];
        px_ele[i] = px[i]*C[i] - py[i]*S[i];
        py[i] = px[i]*S[i] + py[i]*C[i];
        px[i] = px_ele[i];)

        i += stride
        
    end
    s = p_in.s + l
    return nothing
end