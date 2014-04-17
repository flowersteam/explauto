from numpy import pi, array

m_ndims = 8

test_config = dict(m_mins=array(m_ndims*[-0.25]),
                   m_maxs=array(m_ndims*[0.25]),
                   s_mins=array([-pi, -2.5]),
                   s_maxs=array([pi, 2.5]),
                   noise=0.01)
