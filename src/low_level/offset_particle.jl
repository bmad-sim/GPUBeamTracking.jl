using CUDA, BenchmarkTools, Adapt
include("structures.jl")

Adapt.@adapt_structure Particle; Adapt.@adapt_structure Intermediate_Set;
Adapt.@adapt_structure Intermediate_Unset; Adapt.Adapt.@adapt_structure offset_and_tilt;

function offset_particle_set(corrections, p_lab, int)
    """Transform from lab to element coords.
    See Bmad manual (2022-11-06) sections 5.6.1, 15.3.1 and 24.2
    **NOTE**: transverse only as of now.
    """

    x_ele, y_ele, px_ele, py_ele, s, c = int.x_ele, int.y_ele, int.px_ele, int.py_ele, int.s, int.c
    x, px, y, py = p_lab.x, p_lab.px, p_lab.y, p_lab.py
    x_offset, y_offset, tilt = corrections.x_offset, corrections.y_offset, corrections.tilt


    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    i = index
    while i <= length(x)

        @inbounds (s[i] = sin(tilt[i]);
        c[i] = cos(tilt[i]);
        x_ele[i] = x[i] - x_offset[i];
        y_ele[i] = y[i] - y_offset[i];
        x_ele[i] = x_ele[i]*c[i]+ y_ele[i]*s[i];
        y_ele[i] = -x_ele[i]*s[i] + y_ele[i]*c[i];
        px_ele[i] = px[i]*c[i] + py[i]*s[i];
        py_ele[i] = -px[i]*s[i] + py[i]*c[i];)
       
        i += stride
    end

    return
end

# 1000 samples for GPU, 2617 samples for CPU; median times; random values for x_offset, y_offset, and tilt y; 50000 particles
# GPU implementation yields 42.071 μs
# identical CPU implementation: 1.790 ms = 1790 μs
# speedup of ≈ 42.5x 

# 100_000 particles:
# GPU: 86.516 μs, 1000 samples; CPU: 3.638 ms = 3638 μs, 1322 samples
# speedup ≈ 42x

function offset_particle_unset(corrections, p_ele, int)
    """Transforms from element body to lab coords.
    See Bmad manual (2022-11-06) sections 5.6.1, 15.3.1 and 24.2
    **NOTE**: transverse only as of now.
    """

    x_lab, y_lab, px_lab, py_lab, s, c = int.x_lab, int.y_lab, int.px_lab, int.py_lab, int.s, int.c
    x, px, y, py = p_ele.x, p_ele.px, p_ele.y, p_ele.py
    x_offset, y_offset, tilt = corrections.x_offset, corrections.y_offset, corrections.tilt

    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    i = index
    while i <= length(x)

        @inbounds (s[i] = sin(tilt[i]);
        c[i] = cos(tilt[i]);
        x_lab[i] = x[i]*c[i] - y[i]*s[i];
        y_lab[i] = x[i]*s[i] + y[i]*c[i];
        x_lab[i] += x_offset[i];
        y_lab[i] += y_offset[i];
        px_lab[i] = px[i]*c[i] - py[i]*s[i];
        py_lab[i] = px[i]*s[i] + py[i]*c[i];)
        
        i += stride 

    end
    return
end

# incorrect outputs when returning to lab frame