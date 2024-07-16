using CUDA

function optimal_set("""enter function args""")
    """config recommended blocks and threads for function/elements"""
    kernel = @cuda launch=false """function name here"""("""args""")
    config = launch_configuration(kernel.fun)
    threads = min(config.threads, """elements""")
    blocks = cld(threads, """elements""")
end

