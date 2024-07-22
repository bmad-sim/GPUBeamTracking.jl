using CUDA
include("structures.jl");

function fill_quadrupole(elements)
    """filling intermediate calculation arrays for track 
a quadrupole with zeros to avoid dynamic memory allocation; 
change element count as needed."""
    x_ele, px_ele, S, C, x_lab, y_lab, px_lab, py_lab, beta, beta0, e_tot, 
    evaluation, dz, sqrt_k, sk_l, sx, ax11, ax12, ax21, ay11, ay12, ay21,
    cx1, cx2, cx3, cy1, cy2, cy3, b1, rel_p = (CUDA.fill(0.0, elements) for item = 1:30)
    
    return nothing
end
