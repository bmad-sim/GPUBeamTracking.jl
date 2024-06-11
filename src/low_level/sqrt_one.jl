function sqrt_one(x)
    #Routine to calculate Sqrt[1+x] - 1 to machine precision.
    
    sq = sqrt(1 + x)
    rad = sq + 1

    return x/rad

end