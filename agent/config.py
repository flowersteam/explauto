from model import sm_model, i_model
from numpy import array

def get_config(m_ndims, s_ndims, sensorimotor, interest):
    m_dims = range(m_ndims)
    s_dims = range(-s_ndims, 0)
    if sensorimotor[0] == 'imle':
        smm = sm_model.ImleModel(m_dims, s_dims, **sensorimotor[1])
    elif sensorimotor[0] == 'discrete':
        smm = sm_model.LidstoneModel(**sensorimotor[1])
    else:
        print sensorimotor[0], ' is not a valid sensorimotor model'
        raise
    if interest[1] == 'goal':
        i_dims = s_dims
        inf_dims = m_dims
    elif interest[1] == 'motor':
        i_dims = m_dims
        inf_dims = s_dims
    else:
        print interest[1], ' is not a valid interest space'
        raise
    interest[2]['i_dims'] = i_dims
    if interest[0] == 'random':
        im = i_model.RandomInterest(**interest[2])
    elif interest[0] == 'discrete_progress':
        im = i_model.DiscreteProgressInterest(**interest[2])
    else:
        print interest[0], ' is not a valid interest model'
        raise

    return {
            'm_dims' : m_dims,
            's_dims' : s_dims,
            'i_dims' : i_dims,
            'inf_dims' : inf_dims,
            'sm_model' : smm,
            'i_model' : im
            #'competence' : competence
            }
