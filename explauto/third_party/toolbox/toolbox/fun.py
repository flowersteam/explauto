import math, collections, types, os, random

def flatten(l):
    """Flatten any imbrication of iterable collections"""
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, basestring):
            for sub in flatten(el):
                yield sub
        else:
            yield el

def flattenLists(l):
    """Flatten only lists"""
    for el in l:
        #print type(el)
        if type(el) == types.ListType:
            #print "bla"
            for sub in flattenLists(el):
                yield sub
        else:
            yield el

def mapmap(m, f):
    if isinstance(m, collections.Iterable) and not isinstance(m, basestring):
        return [mapmap(e, f) for e in m]
    else:
        return f(m)

def clip(x, a, b):
    """Return the nearest point of x in the interval [a, b]"""
    if a > b:
        return None
    return min(max(x, a), b)

def norm(p1):
    return math.sqrt(sum(p1_i**2 for p1_i in p1))

def dist(p1, p2):
    return math.sqrt(sum((p1i-p2i)**2 for p1i, p2i in zip(p1, p2)))


def dist_sq(p1, p2):
    return sum((p1i-p2i)**2 for p1i, p2i in zip(p1, p2))


def dist_sqw(p1, p2, w = None):
    if w is None:
        return norm_sq(p1, p2)
    else:
        assert len(p1) == len(p2) == len(w)
        return sum((w_i*(p1i-p2i))**2 for w_i, p1i, p2i in zip(w, p1, p2))

def gaussian_kernel(d, sigma_sq):
    """Compute the guassian kernel function of a given distance
    @param d         the euclidean distance
    @param sigma_sq  sigma of the guassian, squared.
    """
    return math.exp(-(d*d)/(2*sigma_sq))

def roulette_wheel(proba):
    assert len(proba) >= 1
    """Given a vector p, return index i with probability p_i/sum(p).
    Elements of p are positive numbers.
    @param proba    list of positive numbers
    """
    sum_proba = sum(proba)
    dice = random.uniform(0., sum_proba)
    if sum_proba == 0.0:
        return random.randint(0, len(proba)-1)
    s, i = proba[0], 0
    while (i < len(proba)-1 and dice >= s):
        i += 1
        assert proba[i] >= 0, "all elements are not positive {}".format(proba)
        s += proba[i]
    return i

