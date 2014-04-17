import numpy as np


def ellipsoid_3d(M, C):
    L, U = np.linalg.eig(C)
    N = 1.5
    radii = N*np.sqrt(L)

    # generate data for "unrotated" ellipsoid
    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    [xc, yc, zc] = [x, y, z]

    # rotate data with orientation matrix U and center M
    a = np.kron(U[:, 0], xc).T
    b = np.kron(U[:, 1], yc).T
    c = np.kron(U[:, 2], zc).T
    data = a+b+c
    n = data.shape[1]
    x = data[0:n, :]+M[0]
    y = data[n:2*n, :]+M[1]
    z = data[2*n:, :]+M[2]

    return x, y, z

# % now plot the rotated ellipse
# %sc = surf(x,y,z);
# %shading interp
# %title('actual ellipsoid represented by data: C and M')
# %axis equal
# %alpha(0.5)
