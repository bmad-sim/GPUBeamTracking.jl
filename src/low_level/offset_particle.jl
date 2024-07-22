using CUDA
include("structures.jl"); include("int_arrays.jl");


function offset_particle_set(corrections, p_lab, int)
    """Transform from lab to element coords.
    See Bmad manual (2022-11-06) sections 5.6.1, 15.3.1 and 24.2
    **NOTE**: transverse only as of now.
    """
    x_ele, px_ele, S, C = int.x_ele, int.px_ele, int.S, int.C

    x, px, y, py = p_lab.x, p_lab.px, p_lab.y, p_lab.py

    x_offset, y_offset, tilt = corrections.x_offset, 
    corrections.y_offset, corrections.tilt

    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    i = index
    while i <= length(x)

        @inbounds (S[i] = sin(tilt[i]);
        C[i] = cos(tilt[i]);
        x[i] -= x_off[i];
        y[i] -= y_off[i];
        x_ele[i] = x[i]*C[i] + y[i]*S[i]; 
        y[i] = -x[i]*S[i] + y[i]*C[i];
        px_ele[i] = px[i]*C[i] + py[i]*S[i];
        py[i] *= C[i];
        py[i] -= px[i]*S[i];)
       
        i += stride
    end
    return nothing
end

function offset_particle_unset(corrections, p_ele, int)
    """Transforms from element bodies to lab coords.
    See Bmad manual (2022-11-06) sections 5.6.1, 15.3.1 and 24.2
    **NOTE**: transverse only as of now.
    """
    x_lab, y_lab, px_lab, py_lab, S, C = int.x_lab, 
    int.y_lab, int.px_lab, int.py_lab, int.S, int.C
    
    x, px, y, py = p_ele.x, p_ele.px, p_ele.y, p_ele.py
    
    x_offset, y_offset, tilt = corrections.x_offset, 
    corrections.y_offset, corrections.tilt

    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    i = index
    while i <= length(x)

        @inbounds (S[i] = sin(tilt[i]);
        C[i] = cos(tilt[i]);
        x_lab[i] = x[i]*C[i] - y[i]*S[i];
        y_lab[i] = x[i]*S[i] + y[i]*C[i];
        x_lab[i] = x_lab[i] + x_offset[i];
        y_lab[i] = y_lab[i] + y_offset[i];
        px_lab[i] = px[i]*C[i] - py[i]*S[i];
        py_lab[i] = px[i]*S[i] + py[i]*C[i];)
        
        i += stride 
    
    end
    return nothing
end

