using CUDA, BenchmarkTools, Adapt
include("low_level/sqrt_one.jl")
include("low_level/structures.jl")

Adapt.@adapt_structure GPU_Particle
Adapt.@adapt_structure GPU_Drift
Adapt.@adapt_structure Intermediate
    
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
    P, Px, Py, Pxy2, Pl, X, Y, dz, dz1, dz2, dz3 = inter.P, inter.Px, inter.Py, 
    inter.Pxy2, inter.Pl, inter.X, inter.Y, inter.dz, inter.dz1, inter.dz2, inter.dz3
    
    @inbounds P .= pz .+ 1
    @inbounds Px .= px ./ P
    @inbounds Py .= py ./ P
    @inbounds Pxy2 .= Px.^2 .+ Py.^2
    @inbounds Pl .= sqrt.(1 .- Pxy2)

    @inbounds X .= L .* Px; Y .= L .* Py
    @inbounds X ./= Pl; Y ./= Pl
    @inbounds x .+= X; y .+= Y

    # z = z + L * ( beta/beta_ref - 1.0/Pl ) but numerically accurate:


    @inbounds dz1 .= 2 .*pz .+ pz.^2; dz1 .*= mc2.^2
    @inbounds dz2 .= p0c.*P; dz2 .^= 2; dz2 .+= mc2.^2
    @inbounds dz3 .= sqrt_one.(.-Pxy2); dz3 ./= Pl
    @inbounds dz .= dz1 ./ dz2
    @inbounds dz .= sqrt_one.(dz); dz .+= dz3; dz .*= L

    @inbounds z .+= dz; s .+= L
    
    

    return
end

"""helper arrays to determine array memory allocation"""
P = CUDA.fill(0.0, 1000); Px = CUDA.fill(0.0, 1000); Py = CUDA.fill(0.0, 1000);
Pxy2 = CUDA.fill(0.0, 1000); Pl = CUDA.fill(0.0, 1000); dz = CUDA.fill(0.0, 1000);
dz1 = CUDA.fill(0.0, 1000); dz2 = CUDA.fill(0.0, 1000); dz3 = CUDA.fill(0.0, 1000);
X = CUDA.fill(0.0, 1000); Y = CUDA.fill(0.0, 1000);

"""sample input of 1000 particles"""
x = CUDA.fill(1.0, 1000); y = CUDA.fill(1.0, 1000); z = CUDA.fill(0.75, 1000);
px = CUDA.fill(0.5, 1000); py = CUDA.fill(0.2, 1000); pz = CUDA.fill(0.5, 1000);
s = CUDA.fill(1.0, 1000); p0c = CUDA.fill(1.0, 1000); mc2 = CUDA.fill(.511, 1000);

L = CUDA.fill(0.005, 1000);


drift = GPU_Drift(L)
inter = Intermediate(P, Px, Py, Pxy2, Pl, X, Y, dz, dz1, dz2, dz3)
p_in = GPU_Particle(x, px, y, py, z, pz, s, p0c, mc2)

function bench_gpu1!(p_in, drift, inter)
    CUDA.@sync begin
    @cuda track_a_drift_gpu!(p_in, drift, inter)
    end
end

@btime bench_gpu1!($p_in, $drift, $inter)
bench_gpu1!(p_in, drift, inter)    # 11.407 ms, 11.437 ms
CUDA.@profile bench_gpu1!(p_in, drift, inter)

function track_a_drift_gpu2!(p_in, drift, inter)
    """Tracks the incoming Particle p_in though drift element
    and returns the outgoing particle. 
    See Bmad manual section 24.9 
    """
    L = drift.L
    s = p_in.s
    p0c = p_in.p0c
    mc2 = p_in.mc2

    x, px, y, py, z, pz = p_in.x, p_in.px, p_in.y, p_in.py, p_in.z, p_in.pz
    P, Px, Py, Pxy2, Pl, X, Y, dz, dz1, dz2, dz3 = inter.P, inter.Px, inter.Py, 
    inter.Pxy2, inter.Pl, inter.X, inter.Y, inter.dz, inter.dz1, inter.dz2, inter.dz3
    
    index = threadIdx().x
    stride = blockDim().x

    for i=index:stride:length(x)
        @inbounds P[i] = pz[i] + 1
        @inbounds Px[i] = px[i] / P[i]
        @inbounds Py[i] = py[i] / P[i]
        @inbounds Pxy2[i] = Px[i]^2 + Py[i]^2
        @inbounds Pl[i] = sqrt(1 - Pxy2[i])

        @inbounds X[i]= L[i]* Px[i]; Y[i]= L[i]* Py[i]
        @inbounds X[i]/= Pl[i]; Y[i]/= Pl[i]
        @inbounds x[i] += X[i]; y[i] += Y[i]

        # z = z + L * ( beta/beta_ref - 1.0/Pl ) but numerically accurate:


        @inbounds dz1[i] = 2 *pz[i] + pz[i]^2; dz1[i] *= mc2[i]^2
        @inbounds dz2[i] = p0c[i]*P[i]; dz2[i] ^= 2; dz2[i] += mc2[i]^2
        @inbounds dz3[i] = sqrt_one(-Pxy2[i]); dz3[i] /= Pl[i]
        @inbounds dz[i] = dz1[i] / dz2[i]
        @inbounds dz[i] = sqrt_one(dz[i]); dz[i] += dz3[i]; dz[i]*= L[i]

        @inbounds z[i] += dz[i]; s[i] += L[i]

    end

    
    

    return
end

@cuda threads=512 track_a_drift_gpu2!(p_in, drift, inter)

function bench_gpu2!(p_in, drift, inter)
    CUDA.@sync begin
    @cuda threads = 500 track_a_drift_gpu2!(p_in, drift, inter)
    end
end

@btime bench_gpu2!($p_in, $drift, $inter)  # 117.152 μs, 117.384 μs