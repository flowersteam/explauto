import testenv

import random
random.seed(20)
import numpy as np
np.random.seed(20)

import models

def PandC_datasets_test():
    """Test if pDataset and cDataset work exactly the same"""
    try:
        from models.dataset import cDataset
        from models.dataset import pDataset
    except:
        print("Test not run, could not import cDataset or pDataset")
        return False

    check = True

    size = 5

    for i in range(100):
        n = random.randint(1, size)
        m = random.randint(1, size)
        cdset = cDataset(n, m)
        pdset = pDataset(n, m)

        for i in range(10):
            x = np.random.rand(n)
            y = np.random.rand(m)
            cdset.add_xy(x, y)
            pdset.add_xy(x, y)
            checkpoint = cdset.size == pdset.size == i+1
            if not checkpoint:
                print('checkpoint 1')
            check = check and checkpoint

        for i in range(10):
            assert tuple(pdset.get_x(i)) == tuple(pdset.get_x(i))

        for i in range(100):

            x = np.random.rand(n)
            k = random.randint(1, size)
            cdists, cnn_x = cdset.nn_x(x, k = k)
            pdists, dnn_x = pdset.nn_x(x, k = k)
            if sorted(cnn_x) != sorted(dnn_x):
                print(n, m, k)
                print(cnn_x, cdists)
                print(dnn_x, pdists*pdists)
                print('Error', x)
                print('c++   ', sorted(cnn_x), cdset.get_x(cnn_x[0]), x - cdset.get_x(cnn_x[0]))
                print('python', sorted(dnn_x), pdset.get_x(dnn_x[1]), x - pdset.get_x(dnn_x[1]))
                pass
            check = check and sorted(cnn_x) == sorted(dnn_x)

            y = np.random.rand(m)
            cnn_y = cdset.nn_y(y, k = k)[1]
            dnn_y = pdset.nn_y(y, k = k)[1]
            check = check and sorted(cnn_y) == sorted(dnn_y)

        # for i in xrange(100):
        #     index = random.randint(0, cdset.size-1)
        #     check = check and np.allclose(cdset.get_x(index), pdset.get_x(index), rtol = 1e-10, atol = 1e-10)
        #     check = check and np.allclose(cdset.get_y(index), pdset.get_y(index), rtol = 1e-10, atol = 1e-10)
        #     check = check and np.allclose(cdset.get_x_padded(index), pdset.get_x_padded(index), rtol = 1e-10, atol = 1e-10)

    return check

def PandC_datasets_test2():
    """Test if pDataset and cDataset work exactly the same, interweaving adding and nn request"""
    try:
        from models.dataset import cDataset
        from models.dataset import pDataset
    except:
        print("Test not run, could not import cDataset or pDataset")
        return False

    check = True

    for i in range(100):
        n = random.randint(1, 20)
        m = random.randint(1, 20)
        n, m = 1, 1
        cdset = cDataset(n, m)
        pdset = pDataset(n, m)

        for i in range(10):
            for j in range(10):
                x = np.random.rand(n)
                y = np.random.rand(m)
                cdset.add_xy(x, y)
                pdset.add_xy(x, y)

                check = check and cdset.size == pdset.size == 10*i+j+1

            for j in range(3):

                x = np.random.rand(n)
                k = random.randint(1, 20)
                cdists, cnn_x = cdset.nn_x(x, k = k)
                pdists, dnn_x = pdset.nn_x(x, k = k)
                if sorted(cnn_x) != sorted(dnn_x):
                    # print cnn_x, cdists
                    # print dnn_x, pdists
                    # print dnn_x, pdists*pdists
                    # print 'Error', x
                    # print 'c++   ', sorted(cnn_x), cdset.get_x(cnn_x[0])[0], x - cdset.get_x(cnn_x[0])[0]
                    # print 'python', sorted(dnn_x), pdset.get_x(dnn_x[0])[0], x - pdset.get_x(dnn_x[0])[0]
                    pass
                check = check and sorted(cnn_x) == sorted(dnn_x)

                y = np.random.rand(m)
                cnn_y = cdset.nn_y(y, k = k)[1]
                dnn_y = pdset.nn_y(y, k = k)[1]
                if sorted(cnn_y) != sorted(dnn_y):
                    # print 'error'
                    # print cnn_y
                    # print dnn_y
                    pass
                check = check and sorted(cnn_y) == sorted(dnn_y)

        for i in range(100):
            index = random.randint(0, cdset.size-1)
            check = check and np.allclose(cdset.get_x(index), pdset.get_x(index), rtol = 1e-10, atol = 1e-10)
            check = check and np.allclose(cdset.get_y(index), pdset.get_y(index), rtol = 1e-10, atol = 1e-10)
            check = check and np.allclose(cdset.get_x_padded(index), pdset.get_x_padded(index), rtol = 1e-10, atol = 1e-10)

        return check

tests = [PandC_datasets_test,
         PandC_datasets_test2
        ]

if __name__ == "__main__":
    print(("\033[1m%s\033[0m" % (__file__,)))
    for t in tests:
        print(('%s %s' % ('\033[1;32mPASS\033[0m' if t() else
                         '\033[1;31mFAIL\033[0m', t.__doc__)))