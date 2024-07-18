using CUDA
include("structures.jl")

function quad_mat2_calc(input, int)
    """Returns 2x2 transfer matrix elements aij and the
    coefficients to calculate the change in z position.
    Input: 
        k1_ref -- Quad strength: k1 > 0 ==> defocus
        len -- Quad length
        rel_p -- Relative momentum P/P0
    Output:
        a11, a12, a21, a22 -- transfer matrix elements
        c1, c2, c3 -- second order derivatives of z such that 
                    z = c1 * x_0^2 + c2 * x_0 * px_0 + c3* px_0^2
    **NOTE**: accumulated error due to machine epsilon. REVISIT
    """ 
    eps = 2.220446049250313e-16  # machine epsilon to double precision
    
    """creating memory copies of arrays"""
    sqrt_k, sk_l, cx, sx, a11, a12, a21, a22, c1, c2, c3 = int.sqrt_k, int.sk_l,
    int.cx, int.sx, int.a11, int.a12, int.a21, int.a22, int.c1, int.c2, int.c3

    k1, l, rel_p = input.k1, input.len, input.rel_p
    
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    i = index
    while i <= length(rel_p)
        sqrt_k[i] = sqrt(abs(k1)+eps)
        sk_l[i] = sqrt_k[i] * len
        
        cx[i] = cos(sk_l[i]) * (k1[i]<=0) + cosh(sk_l[i]) * (k1[i]>0) 
        sx[i] = (sin(sk_l[i])/(sqrt_k[i]))*(k1[i]<=0) + (sinh(sk_l[i])/(sqrt_k[i]))*(k1[i]>0)
            
        a11[i] = cx[i]
        a12[i] = sx[i] / rel_p[i]
        a21[i] = k1[i] * sx[i] * rel_p[i]
        a22[i] = cx[i]
            
        c1[i] = k1[i] * (-cx[i] * sx[i] + l) / 4
        c2[i] = -k1[i] * sx[i]^2 / (2 * rel_p[i])
        c3[i] = -(cx[i] * sx[i] + l) / (4 * rel_p[i]^2)

        i += stride
    end
    return
end
