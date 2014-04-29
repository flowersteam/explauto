"""
In this example we test the impact of sigma on predictions.
"""

import numpy as np
import matplotlib.pyplot as plt

from models.forward import LWLRForwardModel, ESLWLRForwardModel


# noisy sinus samples
def f(x):
    return (x-5)**2*np.sin(x)

x = 10*np.random.rand(100)
y = f(x) + 1.5*np.random.rand(100)

plt.clf()
plt.subplot(221)
plt.plot(x,y,'.')

# prediction with sigma_sq=1.0
model = LWLRForwardModel(1, 1, 1.0, 20)
for i in range(100):
    model.add_xy([x[i]],[y[i]])
x_t = np.arange(-10.,20.,0.01)    # test set
y_t = [model.predict_y([x_t[i]]) for i in range(x_t.shape[0])]
plt.subplot(222)
plt.plot(x_t,y_t)

# prediction with sigma_sq=0.1
model = LWLRForwardModel(1, 1, 0.1, 20)
for i in range(100):
    model.add_xy([x[i]],[y[i]])
y_t=[model.predict_y([x_t[i]]) for i in range(x_t.shape[0])]
plt.subplot(223)
plt.plot(x_t,y_t)

# prediction with estimated_sigma
model = ESLWLRForwardModel(1, 1, 0.1, 20)
for i in range(100):
    model.add_xy([x[i]],[y[i]])
y_t=[model.predict_y([x_t[i]]) for i in range(x_t.shape[0])]
plt.subplot(224)
plt.plot(x_t,y_t)


plt.show()
