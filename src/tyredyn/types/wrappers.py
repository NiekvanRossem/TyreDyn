

def time_func(func):
    """Adds timing to a function call."""

    def wrap_func(*args, **kwargs):

        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()

        print(f"Elapsed time for {func.__name__!r}: {(t2 - t1):.4f} seconds")

        return result

    return wrap_func

