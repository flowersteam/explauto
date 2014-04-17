#include "_cDatasetFlann.h"

int main() {
    
    _cDatasetFlann* cdf = new _cDatasetFlann(3, 2);
    
    for(int i = 0; i < 1000; i++) {
        double x[3] = {i, 2*i, 3*i + 1};
        double y[2] = {-i, 4*i};
        cdf->add_xy(x, y);
    }
    

    for(int i = 0; i < 10000; i++) {
        vector<int>    index;
        vector<double> dists;
        double xq[3] = {i*0.9, 2*i*0.9, 3*i*0.9 + 1};
        cdf->nn_x(20, xq, dists, index);
    }
    
    return 0;
}