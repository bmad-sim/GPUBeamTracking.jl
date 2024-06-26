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

    P .= pz .+ 1;
    Px .= px ./ P;
    Py .= py ./ P;
    Pxy2 .= Px.^2 .+ Py.^2;
    Pl .= sqrt.(1 .- Pxy2);

    x .+= L .* Px  # returns the call to print string error when the function is called



  
    
    z .+= dz;
    s .+= L;

    return
end

P = CUDA.fill(0.0, 10000); Px = CUDA.fill(0.0, 10000); Py = CUDA.fill(0.0, 10000);
Pxy2 = CUDA.fill(0.0, 10000); Pl = CUDA.fill(0.0, 10000); dz = CUDA.fill(0.0, 10000);

"""sample input of 1000 particles"""
x = CUDA.fill(1.0, 10000); y = CUDA.fill(1.0, 10000); z = CUDA.fill(0.75, 10000);
px = CUDA.fill(0.5, 10000); py = CUDA.fill(0.2, 10000); pz = CUDA.fill(0.5, 10000);
s = CUDA.fill(1.0, 10000); p0c = CUDA.fill(1.0, 10000); mc2 = CUDA.fill(.511, 10000);

L = CUDA.fill(0.005, 10000);

drift = GPU_Drift(L);
inter = Intermediate(P, Px, Py, Pxy2, Pl, dz);
p_in = GPU_Particle(x, px, y, py, z, pz, s, p0c, mc2);


@cuda track_a_drift_gpu!(p_in, drift, inter)