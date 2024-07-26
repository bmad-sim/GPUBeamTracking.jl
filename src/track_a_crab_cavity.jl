using CUDA, BenchmarkTools
include("low_level/structures.jl"); include("low_level/int_arrays.jl");
include("low_level/constants.jl"); include("low_level/sqrt_one.jl");

function track_a_crab_cavity!(p_in, cav, int)
    """Tracks an incomming Particle p_in through crab cavity and
    returns the ourgoing particle. 
    See Bmad manual section 4.9
    """
    s = p_in.s
    p0c = p_in.p0c
    mc2 = p_in.mc2
    
    l = cav.L
    
    x_off = cav.X_OFFSET
    y_off = cav.Y_OFFSET
    tilt = cav.TILT

    voltage = cav.VOLTAGE
    rf_freq = cav.RF_FREQUENCY
    k_rf = 2 * pi * rf_freq / c_0;

    phi0 = cav.PHI0
    voltage = cav.VOLTAGE

    x_ele, px_ele, S, C, P, Px, Py, Pxy2, Pl, phase,
    beta, time, E, pc, dz = int.x_ele, int.px_ele, int.S, int.C,
    int.P, int.Px, int.Py, int.Pxy2, int.Pl, int.phase,
    int.beta, int.time, int.E, int.pc, int.dz
    
    x, px, y, py, z, pz = p_in.x, p_in.px, p_in.y, p_in.py, p_in.z, p_in.pz
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
        
        # tracking a drift element
        P[i] = pz[i] + 1;              # CuArray of each Particle's total momentum over p0
        Px[i] = px_ele[i] / P[i];                 # CuArray of each Particle's 'x' momentum over p0
        Py[i] = py[i] / P[i];                 # CuArray of each Particle's 'y' momentum over p0
        Pxy2[i] = Px[i]^2 + Py[i]^2;   # CuArray of each Particle's transverse momentum^2 over p0^2
        Pl[i] = sqrt(1 - Pxy2[i]);     # CuArray of each Particle's longitudinal momentum over p0

        x_ele[i] += l/2 * Px[i] / Pl[i]; 
        y[i] += l/2 * Py[i] / Pl[i];

        # z = z + L * ( beta/beta_ref - 1.0/Pl ) but numerically accurate:
        dz[i] = l/2 * (sqrt_one((mc2^2 * (2 *pz[i]+pz[i]^2))/((p0c[i]*P[i])^2 + mc2^2))
        + sqrt_one(-Pxy2[i])/Pl[i]);
        
        z[i] += dz[i];

        beta[i] = (1+pz[i]) * p0c[i] / sqrt(((1+pz[i])*p0c[i])^2 + mc2[i]^2);
        time[i] = -z[i] / (beta[i] * c_0);

        phase[i] = 2 * pi * (phi0[i] - (time[i]*rf_freq));
        voltage[i] /= p0c[i];
        px_ele[i] += voltage[i] * sin(phase[i]);
        
        beta[i] = (1+pz[i]) * p0c[i] / sqrt(((1+pz[i])*p0c[i])^2 + mc2^2);
        z[i] /= beta[i];

        E[i] =  (1+pz[i]) * p0c[i] / beta[i];
        E[i] += voltage[i] * cos(phase[i]) * k_rf * x_ele[i] * p0c[i];
        pc[i] = sqrt(E[i]^2-mc2^2);
        
        
        beta[i] = pc[i] / E[i];
        
        pz[i] = (pc[i] - p0c[i])/p0c[i];    
        z[i] *= beta[i];
        
         # tracking a drift element
        P[i] = pz[i] + 1;              # CuArray of each Particle's total momentum over p0
        Px[i] = px_ele[i] / P[i];                 # CuArray of each Particle's 'x' momentum over p0
        Py[i] = py[i] / P[i];                 # CuArray of each Particle's 'y' momentum over p0
        Pxy2[i] = Px[i]^2 + Py[i]^2;   # CuArray of each Particle's transverse momentum^2 over p0^2
        Pl[i] = sqrt(1 - Pxy2[i]);     # CuArray of each Particle's longitudinal momentum over p0
 
        x_ele[i] += l/2 * Px[i] / Pl[i]; 
        y[i] += l/2 * Py[i] / Pl[i];
 
         # z = z + L * ( beta/beta_ref - 1.0/Pl ) but numerically accurate:
        dz[i] = l/2 * (sqrt_one((mc2^2 * (2 *pz[i]+pz[i]^2))/((p0c[i]*P[i])^2 + mc2^2))
        + sqrt_one(-Pxy2[i])/Pl[i]);
         
        z[i] += dz[i];
        
        # setting particle back to lab frame
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

