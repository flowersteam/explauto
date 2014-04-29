import toolbox.fun as fun

def fun_coverage():
    """Execute all the function of the fun module"""

    fun.flatten([range(10), range(5)])
    fun.flattenLists([range(10), range(5)])
    fun.clip(1.5, 0.0, 1.0)
    fun.norm((0.0, 0.0), (1.0, 1.0))
    fun.norm_sq((0.0, 0.0), (2.0, -1.0))
    fun.gaussian_kernel(1.0, 2.0)
    fun.roulette_wheel(range(10))

    return True

tests = [fun_coverage]

if __name__ == "__main__":
    print("\033[1m%s\033[0m" % (__file__,))
    for t in tests:
        print('%s %s' % ('\033[1;32mPASS\033[0m' if t() else
                         '\033[1;31mFAIL\033[0m', t.__doc__))