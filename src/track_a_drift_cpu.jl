using CUDA, BenchmarkTools, Test, Adapt
include("low_level/sqrt_one.jl")
include("low_level/structures.jl")


    
function track_a_drift(p_in, drift)
    """Tracks the incoming Particle p_in though drift element
    and returns the outgoing particle. 
    See Bmad manual section 24.9 
    """
    L = drift.L
    s = p_in.s
    p0c = p_in.p0c
    mc2 = p_in.mc2

    x, px, y, py, z, pz = p_in.x, p_in.px, p_in.y, p_in.py, p_in.z, p_in.pz

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

    return particle
end
    


x = fill(1.0, 1024)
y = fill(1.0, 1024)
z = fill(0.75, 1024)
px = fill(0.5, 1024)
py = fill(0.2, 1024)
pz = fill(0.5, 1024)
s = fill(1.0, 1024)
p0c = fill(1.0, 1024)
mc2 = fill(.511, 1024)

L = fill(0.005, 1024)

drift = Drift(L)

p_in = Particle(x, px, y, py, z, pz, s, p0c, mc2)

track_a_drift(p_in, drift)
