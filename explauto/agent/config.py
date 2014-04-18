from numpy import array

from ..interest_models import competences
# from .. import sm_model, i_model


def discrete_active_goal_config(m_card=7, s_card=7,
                                lambd=0.01, win_size=10,
                                measure=competences.competence_bool):
    return get_config(1, 1,
                      ['discrete', dict(m_card=m_card, s_card=s_card, lambd=lambd)],
                      ['discrete_progress',
                       'goal',
                       dict(x_card=s_card, win_size=win_size, measure=measure)])


def imle_config(sensorimotor, interest, sigma0, psi0, **kwargs):
    # m_mins, m_maxs, s_mins, s_maxs):

    m_mins, m_maxs, s_mins, s_maxs = [kwargs[attr]
                                      for attr in ['m_mins', 'm_maxs', 's_mins', 's_maxs']]

    bounds = (tuple(((m_mins[d], m_maxs[d]) for d, _ in enumerate(m_mins))) +
              tuple(((s_mins[d], s_maxs[d]) for d, _ in enumerate(s_mins))))

    return get_config(len(m_mins), len(s_mins),
                      ['imle', dict(sigma0=sigma0, psi0=psi0)],
                      [sensorimotor, interest, dict(bounds=array(bounds).T)])


def get_config(m_ndims, s_ndims, sensorimotor, interest):
    m_dims = range(m_ndims)
    s_dims = range(-s_ndims, 0)

    if sensorimotor[0] == 'imle':
        try:
            from ..sensorimotor_models.imle import ImleModel
            smm = ImleModel(m_dims=m_dims, s_dims=s_dims, **sensorimotor[1])
        except:
            print ('cannot import or instanciate ImleModel, '
                   'please check your installation (compilation, path...)')
            raise
    elif sensorimotor[0] == 'discrete':
        from ..sensorimotor_models import discrete
        smm = discrete.LidstoneModel(**sensorimotor[1])
    else:
        raise ValueError('{} is not a valid sensorimotor model'.format(sensorimotor[0]))

    if interest[1] == 'goal':
        i_dims = s_dims
        inf_dims = m_dims
    elif interest[1] == 'motor':
        i_dims = m_dims
        inf_dims = s_dims
    else:
        raise ValueError('{} is not a valid interest space'.format(interest[1]))

    interest[2]['i_dims'] = i_dims

    if interest[0] == 'random':
        from ..interest_models.random import RandomInterest
        im = RandomInterest(**interest[2])
    elif interest[0] == 'discrete_progress':
        from ..interest_models import DiscreteProgressInterest
        im = DiscreteProgressInterest(**interest[2])
    else:
        raise ValueError('{} is not a valid interest model'.format(interest[0]))

    return {'m_dims': m_dims,
            's_dims': s_dims,
            'i_dims': i_dims,
            'inf_dims': inf_dims,
            'sm_model': smm,
            'i_model': im
            # 'competence' : competence
            }
