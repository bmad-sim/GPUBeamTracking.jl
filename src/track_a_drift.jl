using CUDA, BenchmarkTools, Adapt
include("low_level/sqrt_one.jl")
include("low_level/structures.jl")

"""Adapting structures to bitstype"""
Adapt.@adapt_structure GPU_Particle; Adapt.@adapt_structure GPU_Drift; 
Adapt.@adapt_structure Intermediate;

function track_a_drift_gpu!(p_in, drift, inter)
"""Tracks incoming Particles p_in  on a GPU through 
drift elements. See Bmad manual section 24.9 
"""
    L = drift.L
    s = p_in.s
    p0c = p_in.p0c
    mc2 = p_in.mc2

    x, px, y, py, z, pz = p_in.x, p_in.px, p_in.y, p_in.py, p_in.z, p_in.pz
    P, Px, Py, Pxy2, Pl, dz = inter.P, inter.Px, inter.Py, inter.Pxy2, inter.Pl, inter.dz

    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x   # thread index
    stride = gridDim().x * blockDim().x                                # total number of threads on grid
    
    i = index
    while i <= length(x)

        @inbounds (P[i] = pz[i] + 1;   # CuArray of each Particle's total momentum over p0
        Px[i] = px[i] / P[i];          # CuArray of each Particle's 'x' momentum over p0
        Py[i] = py[i] / P[i];          # CuArray of each Particle's 'y' momentum over p0
        Pxy2[i] = Px[i]^2 + Py[i]^2;   # CuArray of each Particle's transverse momentum^2 over p0^2
        Pl[i] = sqrt(1 - Pxy2[i]);     # CuArray of each Particle's longitudinal momentum over p0

        x[i] += L[i] * Px[i] / Pl[i]; 
        y[i] += L[i] * Py[i] / Pl[i];

        # z = z + L * ( beta/beta_ref - 1.0/Pl ) but numerically accurate:
        dz[i] = L[i] * (sqrt_one((mc2[i]^2 * (2 *pz[i]+pz[i]^2))/((p0c[i]*P[i])^2 + mc2[i]^2))
        + sqrt_one(-Pxy2[i])/Pl[i]);
        
        z[i] += dz[i];
        s[i] += L[i];)

        i += stride;
    end   
    return
end


