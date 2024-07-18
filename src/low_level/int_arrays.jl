using CUDA
include("structures.jl")

function fill_quadrupole(elements)
    """filling intermediate calculation arrays for track 
a quadrupole with zeros to avoid dynamic memory allocation; 
change element count as needed."""
    global x_ele_int, x_ele, y_ele, px_ele, py_ele, S, C, 
    x_lab, y_lab, px_lab, py_lab, beta, beta0, e_tot, 
    evaluation, dz, sqrt_k, sk_l, cx, sx, a11, a12, a21,
    a22, c1, c2, c3, b1 = (CUDA.fill(0.0, elements) for item = 1:28)
    
    return nothing
end

