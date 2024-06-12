include("structures.jl")

function offset_particle_set!(x_offset, y_offset, tilt, p_lab)
    """Transform from lab to element coords.
    See Bmad manual (2022-11-06) sections 5.6.1, 15.3.1 and 24.2
    **NOTE**: transverse only as of now.
    """
    
    (x, px, y, py, z, pz) = p_lab.r
    
    s = sin(tilt)
    c = cos(tilt)
    x_ele_int = p_lab.r[1] - x_offset
    y_ele_int = p_lab.r[3] - y_offset
    x_ele = x_ele_int*c + y_ele_int*s
    y_ele = -x_ele_int*s + y_ele_int*c

    p_lab.r[1], p_lab.r[3] = x_ele, y_ele

    px_ele = p_lab.r[2]*c + p_lab.r[4]*s
    py_ele = -p_lab.r[2]*s + p_lab.r[4]*c

    p_ele = p_lab
    p_ele.r = (x_ele, px_ele, y_ele, py_ele, z, pz)

return p_ele

end


"""Test Value"""
p_lab = Particle(1.0, 1.0, 1.0, [1.0, 0.5, 1.0, 0.2, 1.0, 0.5])
offset_particle_set!(0.01, 0.01, 0.01, p_lab)