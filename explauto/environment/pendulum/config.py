from numpy import pi, array

m_ndims = 8
u_min = -0.25
u_max = 0.25
x_min = -pi
x_max = pi
v_min = -2.5
v_max = 2.5

test_config = dict(m_mins=array(m_ndims*[u_min]),
                   m_maxs=array(m_ndims*[u_max]),
                   s_mins=array([-pi, v_min]),
                   s_maxs=array([pi, v_max]),
                   noise=0.0)
