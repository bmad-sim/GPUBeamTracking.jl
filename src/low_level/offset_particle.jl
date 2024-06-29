using CUDA, BenchmarkTools, Adapt
include("structures.jl")

Adapt.@adapt_structure GPU_Particle; Adapt.@adapt_structure Intermediate_Offset;

function offset_particle_set!(x_offset, y_offset, tilt, p_lab, int)
    """Transform from lab to element coords.
    See Bmad manual (2022-11-06) sections 5.6.1, 15.3.1 and 24.2
    **NOTE**: transverse only as of now.
    """

    x_ele_int, y_ele_int, x_ele, y_ele, px_ele, py_ele = int.x_ele_int, 
    int.y_ele_int, int.x_ele, int.y_ele, int.px_ele, int.py_ele
    x, px, y, py = p_lab.x, p_lab.px, p_lab.y, p_lab.py

    index = (blockIdx().x - Float32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    while i <= length(x)
        
        @inbounds (s[i] = sin(tilt[i]);
        c[i] = cos(tilt[i]);
        x_ele_int[i] = x[i] - x_offset[i];
        y_ele_int[i] = y[i] - y_offset[i];
        x_ele[i] = x_ele_int[i]*c[i]+ y_ele_int[i]*s[i];
        y_ele[i] = -x_ele_int[i]*s[i] + y_ele_int[i]*c[i];
        px_ele[i] = px[i]*c[i] + py[i]*s[i];
        py_ele[i] = -px[i]*s[i] + py[i]*c[i];)

        i += stride
    end

    GPU_particles = GPU_Particle(x_ele, px_ele, y_ele, py_ele, z, pz, s, p0c, mc2)

    return
end


"""Test Value"""
x_offset = CUDA.fill(0.01, 50000); y_offset = CUDA.fill(0.01, 50000); tilt = CUDA.fill(0.01, 50000);

p_lab = GPU_Particle(x, px, y, py, z, pz, s, p0c, mc2)
