from numpy import zeros, matrix, array, hstack
from sklearn.preprocessing import MinMaxScaler

from ..models.gmminf import GMM


class DynamicAttractor(object):
    def __init__(self, ndims, n_components, dt, kv, kp, **kwargsGMM):
        self.gmm =  GMM(n_components=n_components, covariance_type='full', **kwargsGMM)
        self.ndims = ndims
        self.kv = kv
        self.kp = kp
        self.dt = dt
        self.scaler = MinMaxScaler(feature_range=(-0.1, 0.1))

    def fit(self, data):
        #scaled_data = self.scaler.fit_transform(data)
        #self.gmm.fit(scaled_data)
        self.gmm.fit(data)

    def next_state(self, pos, spd):
        h = self.gmm.predict_proba(hstack((pos, spd)).reshape(1, -1))
        # print pos[0], spd[0]
        # print h
        next_pos = zeros(self.ndims)
        next_spd = zeros(self.ndims)
        for k, (w, m, c) in enumerate(self.gmm):
            m_pos = matrix(m[:self.ndims].reshape(-1, 1))
            m_spd = matrix(m[-self.ndims:].reshape(-1, 1))
            c_pos = matrix(c[:self.ndims, :self.ndims])
            c_spd = matrix(c[-self.ndims:, -self.ndims:])
            c_pos_spd = matrix(c[:self.ndims, -self.ndims:])
            c_spd_pos = matrix(c[-self.ndims:, :self.ndims])
            next_pos += (h[:, k] * array(m_pos + c_pos_spd * c_spd.I * (spd.reshape(-1, 1) - m_spd))).flatten()
            next_spd += (h[:, k] * array(m_spd + c_spd_pos * c_pos.I * (pos.reshape(-1, 1) - m_pos))).flatten()
        return next_pos, next_spd

    def command(self, pos, spd):
        #pos_spd = self.scaler.transform(hstack((pos_, spd_)))
        #pos = pos_spd[:self.ndims]
        #spd = pos_spd[-self.ndims:]
        des_pos, des_spd = self.next_state(pos, spd)
        acc = (des_spd - spd) * self.kv + (des_pos - pos) * self.kp
        comm_spd = spd + self.dt * acc
        comm_pos = pos + self.dt * comm_spd
        #comm_pos_spd = self.scaler.inverse_transform(hstack((comm_pos, comm_spd)))
        comm_pos_spd = hstack((comm_pos, comm_spd))
        return comm_pos_spd[:self.ndims], comm_pos_spd[-self.ndims:]

