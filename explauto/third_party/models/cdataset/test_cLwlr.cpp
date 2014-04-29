#include "_cDatasetFlann.h"
#include "_cLwlr.h"

int main() {


    for(int k = 0; k < 100; k++) {
        _cDatasetFlann* cdf = new _cDatasetFlann(3, 2);
        
        for(int i = 0; i < 100; i++) {
            double x[3] = {i, 2*i, 3*i + 1};
            double y[2] = {-i, 4*i};
            cdf->add_xy(x, y);
        }
        
        _cLwlr* clwlr = new _cLwlr(3, 2, 4, 5.0, cdf);
        double x[3] = {0.0, 0.0, 0.0};
        double y[2] = {0.0, 0.0};
        clwlr->predict_y(x, y, 4, 5.0);
        
        return 0;    
    }
}