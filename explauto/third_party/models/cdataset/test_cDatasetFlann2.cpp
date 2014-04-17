#include <cassert>
#include <iostream>
#include "_cDataset.h"

using namespace std;

int main() {

    _cDataset* cdf = new _cDataset(3, 2);

    for(int i = 0; i < 1000; i++) {
//        double x[3] = {i, 2*i, 3*i};
        double x[3] = {i,  0.0, 0.0};
        double y[2] = {-i, 4*i};
        cdf->add_xy(x, y);
    }


    for(int i = 0; i < 1000; i++) {
        vector<int>    index;
        vector<double> dists;
        double xq[3] = {i + 0.01, 0.0,  0.0};
//        double xq[3] = {i + 0.01, 2*i - 0.01, 3*i - 0.01};
        cdf->nn_x(1, xq, dists, index);
        std::cout << index[0] << ":" << i << std::endl;
        //assert(index[0] == i);
    }

    return 0;
}