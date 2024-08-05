using CUDA
include("low_level/structures.jl"); include("low_level/int_arrays.jl"); include("low_level/sqrt_one.jl");
include("low_level/constants.jl");

function track_a_rf_cavity!(p_in, cav, int)
    """Tracks incomming particles p_in through rf cavity and
    returns the outgoing particles. 
    See Bmad manual section 4.9
    """
    p0c = p_in.p0c
    mc2 = p_in.mc2

    l = cav.L

    x_off = cav.X_OFFSET
    y_off = cav.Y_OFFSET
    tilt = cav.TILT

    voltage = cav.VOLTAGE
    phi0 = cav.PHI0
    rf_freq = cav.RF_FREQUENCY

    x, px, y, py, z, pz = p_in.x, p_in.px, p_in.y, p_in.py, p_in.z, p_in.pz

    x_ele, px_ele, S, C, phase, pc, beta, E, E_old, dE, time, z_old, dz,
    P, Px, Py, Pxy2, Pl = int.x_ele, int.px_ele, int.S, int.C, int.phase,
    int.pc, int.beta, int.E, int.E_old, int.dE, int.time, int.z_old, int.dz,
    int.P, int.Px, int.Py, int.Pxy2, int.Pl

    # indexing threads and defining kernel grid length (stride)
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    i = index

    while i <= length(x)
        
        # setting to element frame
        @inbounds (
        S[i] = sin(tilt[i]);
        C[i] = cos(tilt[i]);
        x[i] -= x_off[i];
        y[i] -= y_off[i];
        x_ele[i] = x[i]*C[i] + y[i]*S[i]; 
        y[i] = -x[i]*S[i] + y[i]*C[i];
        px_ele[i] = px[i]*C[i] + py[i]*S[i];
        py[i] *= C[i];
        py[i] -= px[i]*S[i];

        # calculating phase
        beta[i] = (1+pz[i]) * p0c[i] / sqrt(((1+pz[i])*p0c[i])^2 + mc2[i]^2);
        time[i] = -z[i] / (beta[i] * c_0); 
        phase[i] = 2 * pi * (phi0[i] - (time[i]*rf_freq));
        
        # energy change dE
        dE[i] = voltage[i] * sin(phase[i]) / 2;

        # apply energy kick
        pc[i] = (1 + pz[i]) * p0c[i];
        beta[i]= (1+pz[i]) * p0c[i] / sqrt(((1+pz[i])*p0c[i])^2 + mc2[i]^2);
        E_old[i] =  pc[i] / beta[i];  # E_old

        E[i] = E_old[i] + dE[i];

        pz[i] += (1 + pz[i]) * sqrt_one((2*E_old[i]*dE[i] + dE[i]^2)/pc[i]^2);
        pc[i] = p0c[i] * (1 + pz[i]);
        z[i] /= beta[i];
        beta[i] = pc[i] / E[i] ; # beta_new
        z[i] *= beta[i];
        
        z_old[i] = z[i];

        P[i] = pz[i] + 1;              # CuArray of each Particle's total momentum over p0
        Px[i] = px_ele[i] / P[i];      # CuArray of each Particle's 'x' momentum over p0
        Py[i] = py[i] / P[i];          # CuArray of each Particle's 'y' momentum over p0
        Pxy2[i] = Px[i]^2 + Py[i]^2;   # CuArray of each Particle's transverse momentum^2 over p0^2
        Pl[i] = sqrt(1 - Pxy2[i]);     # CuArray of each Particle's longitudinal momentum over p0

        x_ele[i] += l * Px[i] / Pl[i]; 
        y[i] += l * Py[i] / Pl[i];

        # z = z + L * ( beta/beta_ref - 1.0/Pl ) but numerically accurate:
        dz[i] = l * (sqrt_one((mc2^2 * (2 *pz[i]+pz[i]^2))/((p0c[i]*P[i])^2 + mc2^2))
        + sqrt_one(-Pxy2[i])/Pl[i]);
        
        z[i] += dz[i];
        
        # apply second energy kick
        beta[i] = (1+pz[i]) * p0c[i] / sqrt(((1+pz[i])*p0c[i])^2 + mc2^2);  # beta new
        phase[i] += 2 * pi * rf_freq * (z[i]-z_old[i])/(c_0*beta[i]);

        dE[i] = voltage[i] * sin(phase[i]) / 2;

        pc[i] = (1 + pz[i]) * p0c[i];
        beta[i]= (1+pz[i]) * p0c[i] / sqrt(((1+pz[i])*p0c[i])^2 + mc2^2);
        E_old[i] =  pc[i] / beta[i];  # E_old

        E[i] = E_old[i] + dE[i];

        pz[i] += (1 + pz[i]) * sqrt_one((2*E_old[i]*dE[i] + dE[i]^2)/pc[i]^2);
        pc[i] = p0c[i] * (1 + pz[i]);
        z[i] /= beta[i];
        beta[i] = pc[i] / E[i] ; # beta_new
        z[i] *= beta[i]; 

        # set frame back to lab
        x[i] = x_ele[i]*C[i] - y[i]*S[i];
        y[i] = x_ele[i]*S[i] + y[i]*C[i];
        x[i] += x_off[i];
        y[i] += y_off[i];
        px[i] = px_ele[i]*C[i] - py[i]*S[i];
        py[i] = px_ele[i]*S[i] + py[i]*C[i]; )

        i += stride
    end
    s = p_in.s + l
    return nothing
end