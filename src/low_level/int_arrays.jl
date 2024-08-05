using CUDA
include("structures.jl");

"""filling intermediate calculation arrays
with zeros as bitstypr to avoid dynamic memory 
allocation;change element count as needed."""

function fill_drift(elements)
    global P, Px, Py, Pxy2, Pl, dz = (CUDA.fill(0.0, elements) for item = 1:6)
    
    return nothing
end

function fill_quadrupole(elements)
    global x_ele, px_ele, y_ele, S, C,
    beta, beta0, e_tot, evaluation, dz, sqrt_k, sk_l, sx, a11, 
    a12, a21, c1, c2, c3, rel_p = (CUDA.fill(0.0, elements) for item = 1:21)
    
    return nothing
end

function fill_sextupole(elements)
    global x_ele, px_ele, y_ele, S, C, b2, beta, beta0,
    e_tot, evaluation, dz, rel_p = (CUDA.fill(0.0, elements) for item = 1:12)

    return nothing
end

function fill_rf(elements)
    global x_ele, px_ele, S, C, phase, pc, beta, E, E_old, dE, time,
    z_old, dz, P, Px, Py, Pxy2, Pl = (CUDA.fill(0.0, elements) for item = 1:18)

    return nothing
end

function fill_crab(elements)
    global x_ele, px_ele, S, C, phase, pc, beta, E, time,
    dz, P, Px, Py, Pxy2, Pl, dz = (CUDA.fill(0.0, elements) for item = 1:16)

    return nothing
end