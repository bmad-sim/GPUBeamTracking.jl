include("structures.jl")
include("low_level/sqrt_one.jl")


function track_a_drift!(p_in, drift) 
    """Tracks the incoming Particle p_in though drift element
    and returns the outgoing particle. 
    See Bmad manual section 24.9"""
    
    L = drift.L
        
    s = p_in.s
    p0c = p_in.p0c
    mc2 = p_in.mc2
        
    (x, px, y, py, z, pz) = p_in.r
        
    P = 1 + pz            # Particle's total momentum over p0
    Px = px / P           # Particle's 'x' momentum over p0
    Py = py / P           # Particle's 'y' momentum over p0
    Pxy2 = Px^2 + Py^2  # Particle's transverse mometum^2 over p0^2
    Pl = sqrt(1-Pxy2)     # Particle's longitudinal momentum over p0
        
    p_in.r[1] = x + L * Px / Pl
    p_in.r[3] = y + L * Py / Pl

    # z = z + L * ( beta/beta_ref - 1.0/Pl ) but numerically accurate:
    dz = L * (sqrt_one((mc2^2 * (2*pz+pz^2))/((p0c*P)^2 + mc2^2))
    + sqrt_one(-Pxy2)/Pl)
    p_in.r[5] = z + dz
    p_in.s = s + L
        

    return p_in
end

