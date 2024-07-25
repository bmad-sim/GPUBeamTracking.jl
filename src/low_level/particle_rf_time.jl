using CUDA, BenchmarkTools
include("structures.jl"); include("constants.jl");


function particle_rf_time(p)
    """Returns rf time of Particle p."""
    beta, time = p.beta, p.time
    
    z, pz, p0c, mc2 = p.z, p.pz, p.p0c, p.mc2

    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    i = index

    while i <= length(z)

        @inbounds ( beta[i] = (1+pz[i]) * p0c[i] / sqrt(((1+pz[i])*p0c[i])^2 + mc2[i]^2);
        time[i] = -z[i] / (beta[i] * c_0); )

        i += stride
    end
    return nothing
end