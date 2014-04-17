#include <cmath>
#include <cassert>
#include <iostream>

#include "predict.h"
#include "_cLwlr.h"

_cLwlr::_cLwlr(int dim_x, int dim_y, int k, double sigma, _cDataset* dset) {
    this->dim_x    = dim_x;
    this->dim_y    = dim_y;
    this->k        = k;
    this->sigma    = sigma;
    this->sigma_sq = sigma*sigma;
    this->dataset  = dset;
    this->es       = false;

    // Allocatting reccurent temporary data objects
    this->_xqext.resize(1+this->dim_x); // [1.0] + xq
    this->_xqext[0] = 1.0;
}

_cLwlr::~_cLwlr() {
    //printf("deallocating cLwlr\n");
    //delete this->dataset;

    //delete this->_w;
    //delete this->_xqext;
    //delete this->_x;
    //delete this->_y;
}

void _cLwlr::predict_y(double xq[], double yq[], int k, double sigma_sq) {

    assert(this->dataset->size >= k);

    vector<int>    index;
    vector<double> dists;
    this->dataset->nn_x_v(k, xq, dists, index);
    this->_weights(k, sigma_sq, dists); // Compute _w

    // Resizing if too small
    int knn = index.size();
    if (knn*(this->dim_x+1) > int(this->_x.size())) {
        this->_x.resize(knn*(this->dim_x+1));
    }
    if (knn*this->dim_y > int(this->_y.size())) {
        this->_y.resize(knn*this->dim_y);
    }

    for (int i = 0; i < knn; i++) {
        this->_x[(this->dim_x+1)*i] = 1.0;
        int idxi = index[i];

        for (int j = 0; j < this->dim_x; j++) {
            this->_x[(this->dim_x+1)*i+j+1] = this->dataset->_data_x[idxi][j];
        }
        for (int j = 0; j < this->dim_y; j++) {
            this->_y[(this->dim_y)*i+j] = this->dataset->_data_y[idxi][j];
        }
    }

    for (int i = 0; i < this->dim_x; i++) {
        this->_xqext[i+1] = xq[i];
    }

    predictLWR(this->dim_x+1, this->dim_y, k, &this->_xqext[0], &this->_x[0], &this->_y[0], &this->_w[0], yq);
}

// k is the number of effective neighbors in dataset->_index and dataset->_dists
void _cLwlr::_weights(int k, double sigma_sq, vector<double>& dists) {

    double wsum = 0.0;

    if (this->es) {
        sigma_sq = 0.0;
        for (int i = 0; i < k; i++) {
            double di = dists[i];
            sigma_sq += di*di;
        }
        sigma_sq /= k;
    }

    if (k > int(this->_w.size())) {
        this->_w.resize(k);
    }
    for (int i = 0; i < k; i++) {
        double g_i = this->_gauss(dists[i], sigma_sq);
        this->_w[i] = g_i;
        wsum       += g_i;
    }

    if (wsum == 0) {
        // Uniform weights
        double w = 1.0/k;
        for (int i = 0; i < k; i++) {
            this->_w[i] = w;
        }
    } else {
        // Eliminating outliers & Normalizing
        double eps = wsum*1e-15/this->dim_x;
        for (int i = 0; i < k; i++) {
            if (this->_w[i] < eps) {
                this->_w[i] = 0.0;
            } else {
                this->_w[i] /= wsum;
            }
        }
    }
}

//const double counterbalance = 10.0e30;

double _cLwlr::_gauss(double a, double sigma_sq) {
    return exp(-sqrt(a)/(sigma_sq));
}
