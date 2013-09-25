import sys
sys.path.append('../build/lib')

import imle

param = imle.ImleParam([0.1] * 3, 1.0,
                       100.0,
                       0.3,
                       0.9)
i = imle.Imle(param)

import numpy

for _ in range(100):
    l1 = list(numpy.random.randn(7))
    l2 = list(numpy.random.randn(3))

    i.update(l1, l2)


print i.predict(range(7))
print i.predict(range(7))
print i.predict(range(7))


print i.predict_inverse([4, 5, 6])