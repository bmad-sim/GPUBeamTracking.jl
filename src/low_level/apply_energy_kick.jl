using CUDA, BenchmarkTools
include("structures.jl"); include("sqrt_one.jl");

function apply_energy_kick!(dE, p_in, int)
"""Changes the energy of a particle by dE."""
    z, pz, p0c, mc2 = p_in.z, p_in.pz, p_in.p0c, p_in.mc2
    
    beta, E, E_old, pc = int.beta, int.E, int.E_old, int.pc

    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    i = index

    while i <= length(z)

        @inbounds ( pc[i] = (1 + pz[i]) * p0c[i];
        beta[i]= (1+pz[i]) * p0c[i] / sqrt(((1+pz[i])*p0c[i])^2 + mc2[i]^2);
        E_old[i] =  pc[i] / beta[i];  # E_old

        E[i] = E_old[i] + dE;

        pz[i] += (1 + pz[i]) * sqrt_one((2*E_old[i]*dE + dE^2)/pc[i]^2);
        pc[i] = p0c[i] * (1 + pz[i]);
        z[i] /= beta[i];
        beta[i] = pc[i] / E[i] ; # beta_new
        z[i] *= beta[i]; )
        
        i += stride
     
    end
    return nothing
end





