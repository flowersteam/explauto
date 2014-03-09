from sm_model import SmModel
from ..utils import discrete_random_draw

class LidstoneModel(SmModel):
    def __init__(self, m_card, s_card, lambd = 1):
        self.k = m_card * s_card
        self.counts = np.zeros((m_card, s_card))
        self.lambd = lambd
        self.n = 0.
    def joint_distr(self):
        return (self.counts + self.lambd) / (self.n + self.k * self.lambd)

    def infer(self, in_dims, out_dims, x):
        if in_dims == 0:
            p_out = self.joint_distr()[x, :]
        else:
            p_out = self.joint_distr()[:, x]
        p_out /= p_out.sum()
        return discrete_random_draw(p_out.flatten())

    def update(self, m, s):
        self.counts[int(m), int(s)] += 1
        self.n += 1
