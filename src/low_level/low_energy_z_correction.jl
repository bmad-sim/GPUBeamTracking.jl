using CUDA
include("structures.jl")

function low_energy_z_correction(z_calc, int)
    """Corrects the change in z-coordinate due to speed < c_light.
    Input:
        p0c -- reference particle momentum in eV
        mass -- particle mass in eV
    Output: 
        dz -- dz=(ds-d_particle) + ds*(beta - beta_ref)/beta_ref
    """
    beta, beta0, e_tot, evaluation, dz = int.beta, int.beta0, int.e_tot, int.evaluation, int.dz
    
    pz, p0c, mass, ds = z_calc.pz, z_calc.p0c, z_calc.mass, z_calc.ds

    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    i = index
    while i <= length(p0c)
        beta[i] = (1+pz[i]) * p0c[i] / sqrt(((1+pz[i])*p0c[i])^2 + mass[i]^2)
        beta0[i] = p0c[i] / sqrt( p0c[i]^2 + mass[i]^2)
        e_tot[i] = sqrt(p0c[i]^2+mass[i]^2)
        
        evaluation[i] = mass[i] * (beta0[i]*pz[i])^2
        dz[i] = (ds[i] * pz[i] * (1 - 3*(pz[i]*beta0[i]^2)/2+pz[i]^2*beta0[i]^2
                * (2*beta0[i]^2-(mass[i]/e_tot[i])^2/2) )
                * (mass[i]/e_tot[i])^2
                * (evaluation[i]<3e-7*e_tot) 
                + (ds[i]*(beta[i]-beta0[i])/beta0[i])
                * (evaluation[i]>=3e-7*e_tot) )
           
        i += stride
    end
    return 
end