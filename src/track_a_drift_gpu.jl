using CUDA, BenchmarkTools, Adapt
include("low_level/sqrt_one.jl")
include("low_level/structures.jl")

"""Adapting structures to bitstype"""
Adapt.@adapt_structure GPU_Particle; Adapt.@adapt_structure GPU_Drift; 
Adapt.@adapt_structure Intermediate;

function track_a_drift_gpu!(p_in, drift, inter)
"""Tracks the incoming Particle p_in though drift element
and returns the outgoing particle. 
See Bmad manual section 24.9 
"""
    L = drift.L
    s = p_in.s
    p0c = p_in.p0c
    mc2 = p_in.mc2

    x, px, y, py, z, pz = p_in.x, p_in.px, p_in.y, p_in.py, p_in.z, p_in.pz
    P, Px, Py, Pxy2, Pl, dz = inter.P, inter.Px, inter.Py, inter.Pxy2, inter.Pl, inter.dz

    @inbounds (P .= pz .+ 1;
    Px .= px ./ P;
    Py .= py ./ P;
    Pxy2 .= Px.^2 .+ Py.^2;
    Pl .= sqrt.(1 .- Pxy2);)

    i = 1
    while i <= length(x)
        
        @inbounds (x[i] += L[i] * Px[i] / Pl[i]; 
        y[i] += L[i] * Py[i] / Pl[i];

        # z = z + L * ( beta/beta_ref - 1.0/Pl ) but numerically accurate:
        dz[i] = L[i] * (sqrt_one((mc2[i]^2 * (2 *pz[i]+pz[i]^2))/((p0c[i]*P[i])^2 + mc2[i]^2))
        + sqrt_one(-Pxy2[i])/Pl[i]))
        i += 1

    end

    @inbounds (z .+= dz;
            s .+= L)

    return
end



