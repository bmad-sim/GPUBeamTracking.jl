using CUDA, BenchmarkTools, Adapt
include("low_level/sqrt_one.jl")
include("low_level/structures.jl")

"""Adapting structures to bitstype"""
Adapt.@adapt_structure GPU_Particle; Adapt.@adapt_structure GPU_Drift; 
Adapt.@adapt_structure Intermediate;

"""helper arrays to determine array memory allocation"""
P = CUDA.fill(0.0, 1000); Px = CUDA.fill(0.0, 1000); Py = CUDA.fill(0.0, 1000);
Pxy2 = CUDA.fill(0.0, 1000); Pl = CUDA.fill(0.0, 1000); dz = CUDA.fill(0.0, 1000);

"""sample input of 1000 particles"""
x = CUDA.fill(1.0, 1000); y = CUDA.fill(1.0, 1000); z = CUDA.fill(0.75, 1000);
px = CUDA.fill(0.5, 1000); py = CUDA.fill(0.2, 1000); pz = CUDA.fill(0.5, 1000);
s = CUDA.fill(1.0, 1000); p0c = CUDA.fill(1.0, 1000); mc2 = CUDA.fill(.511, 1000);

L = CUDA.fill(0.005, 1000);

drift = GPU_Drift(L)
inter = Intermediate(P, Px, Py, Pxy2, Pl, dz)
p_in = GPU_Particle(x, px, y, py, z, pz, s, p0c, mc2)

function track_a_drift_gpu!(p_in, drift, inter)
    """Tracks incoming particles p_in through drift elements
    and returns the outgoing particles on GPU device. 
    See Bmad manual section 24.9 
    """
    L = drift.L
    s = p_in.s
    p0c = p_in.p0c
    mc2 = p_in.mc2

    x, px, y, py, z, pz = p_in.x, p_in.px, p_in.y, p_in.py, p_in.z, p_in.pz
    P, Px, Py, Pxy2, Pl, dz = inter.P, inter.Px, inter.Py, inter.Pxy2, inter.Pl, inter.dz
    
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    for i = index:stride:length(x)
        @inbounds (P[i] = pz[i] + 1;  # Particle's total momentum over p0
        Px[i] = px[i] / P[i];         # Particle's 'x' momentum over p0
        Py[i] = py[i] / P[i];         # Particle's 'y' momentum over p0
        Pxy2[i] = Px[i]^2 + Py[i]^2;  # Particle's transverse mometum^2 over p0^2
        Pl[i] = sqrt(1 - Pxy2[i]);    # Particle's longitudinal momentum over p0

        x[i] += L[i] * Px[i] / Pl[i]; 
        y[i] += L[i] * Py[i] / Pl[i];)

        # z = z + L * ( beta/beta_ref - 1.0/Pl ) but numerically accurate:
        @inbounds (dz[i] = L[i] * (sqrt_one((mc2[i]^2 * (2 *pz[i]+pz[i]^2))/((p0c[i]*P[i])^2 + mc2[i]^2))
        + sqrt_one(-Pxy2[i])/Pl[i]))

        @inbounds (z[i] += dz[i]; 
                s[i] += L[i];)
    end
    
    return
end

"""configuring the number of threads/blocks
to saturate the GPU and benchmarking"""
function config_bench_gpu!(p_in, drift, inter)
    kernel = @cuda launch=false track_a_drift_gpu!(p_in, drift, inter)
    config = launch_configuration(kernel.fun)
    threads = min(length(x), config.threads)
    blocks = cld(length(x), threads)

    CUDA.@sync begin
    kernel(p_in, drift, inter; threads, blocks)
    end
end

@btime config_bench_gpu!($p_in, $drift, $inter) #1000 particles:
                                                # first test -> 60.673 μs; second test -> 60.620 μs

function bench_gpu!(p_in, drift, inter)
    CUDA.@sync begin
    @cuda threads=768 blocks=2 track_a_drift_gpu!(p_in, drift, inter) # configured threads and blocks
    end
end

@btime bench_gpu!($p_in, $drift, $inter) # 1000 particles (without configuration included):
                                         # first test -> 57.336 μs; second test -> 58.054 μs
