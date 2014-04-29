#ifndef _CLWLR_H_
#define _CLWLR_H_

#include "_cDataset.h"

using namespace std;

class _cLwlr {
    public:
        _cLwlr(int dim_x, int dim_y, int k, double sigma, _cDataset* dset);
        ~_cLwlr();
        int dim_x, dim_y;
        int k;
        double sigma, sigma_sq;
        int es;
        _cDataset* dataset;

        void predict_y(double xq[], double yq[], int k, double sigma_sq);

    private:
        // Temporary data
        vector<double>     _w;
        vector<double> _xqext;
        vector<double> _yqext;
        vector<double>     _x;
        vector<double>     _y;

        // Functions

        void _weights(int, double, vector<double>& dists);
        double _gauss(double, double);
};

#endif // _CLWR_H_
