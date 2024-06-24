using CUDA, BenchmarkTools, Test, Adapt
include("low_level/sqrt_one.jl")
include("low_level/structures.jl")



"""Adapting structs to bitstype"""
Adapt.@adapt_structure GPU_Particle
Adapt.@adapt_structure GPU_Drift
Adapt.@adapt_structure Intermediate
    

function track_a_drift_GPU!(p_in, drift, inter)
    """Parallelized tracking of incoming particles p_in 
    through drift elements, computed on the GPU.
    See Bmad manual section 24.9 
    """
    L = drift.L
    s = p_in.s
    p0c = p_in.p0c
    mc2 = p_in.mc2

    x, px, y, py, z, pz = p_in.x, p_in.px, p_in.y, p_in.py, p_in.z, p_in.pz
    P, Px, Py, Pxy2, Pl, dz = inter.P, inter.Px, inter.Py, inter.Pxy2, inter.Pl

    P = 1 .+ pz            # Particle's total momentum over p0
    Px = px ./ P           # Particle's 'x' momentum over p0
    Py = py ./ P           # Particle's 'y' momentum over p0
    Pxy2 = Px.^2 + Py.^2  # Particle's transverse mometum^2 over p0^2
    Pl = sqrt.(1 .- Pxy2)     # Particle's longitudinal momentum over p0
    x .+= L .* Px ./ Pl
    y .+= L .* Py ./ Pl
    
    # z = z + L * ( beta/beta_ref - 1.0/Pl ) but numerically accurate:
    dz = L .* (sqrt_one.((mc2.^2 .* (2 .*pz.+pz.^2))./((p0c.*P).^2 .+ mc2.^2))
                .+ sqrt_one.(-Pxy2)./Pl)
    
    z .+= dz
    s .+= L
    particle = p_in
    
    return 
end


    
"""helper arrays to determine array memory allocation"""
P = CUDA.fill(undef, 1000);
Px = CUDA.fill(undef, 1000);
Py = CUDA.fill(undef, 1000);
Pxy2 = CUDA.fill(undef, 1000);
Pl = CUDA.fill(undef, 1000);
dz = CUDA.fill(undef, 1000);

"""sample input of 1000 particles"""
x = CUDA.fill(1.0, 1000);
y = CUDA.fill(1.0, 1000);
z = CUDA.fill(0.75, 1000);
px = CUDA.fill(0.5, 1000);
py = CUDA.fill(0.2, 1000);
pz = CUDA.fill(0.5, 1000);
s = CUDA.fill(1.0, 1000);
p0c = CUDA.fill(1.0, 1000);
mc2 = CUDA.fill(.511, 1000);

L = CUDA.fill(0.005, 1000);

drift = GPU_Drift(L)
inter = Intermediate(P, Px, Py, Pxy2, Pl, dz)
p_in = GPU_Particle(x, px, y, py, z, pz, s, p0c, mc2)

"""benchmarking"""
function bench_track_a_drift(p_in, drift, inter)
    kernel = @cuda launch=false track_a_drift_GPU!(p_in, drift, inter)
    config = launch_configuration(kernel.fun)
    threads = config.threads
    blocks = config.blocks

    CUDA.@sync begin
        kernel(p_in, drift, inter; threads, blocks)
    end
end

@btime bench_track_a_drift($p_in, $drift, $inter)
