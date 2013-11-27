#include <cmath>
#include <cassert>
#include <iostream>

#include "_cDatasetFlann.h"

const double EPS = 0.000000001;

_cDatasetFlann::_cDatasetFlann(int dim_x, int dim_y) {
    this->dim_x = dim_x;
    this->dim_y = dim_y;
    this->size = 0;

    this->_index_x = NULL;
    this->_index_y = NULL;
}

_cDatasetFlann::~_cDatasetFlann() {

}

void _cDatasetFlann::reset() {
    this->_data_x.clear();
    this->_data_y.clear();
    this->size = 0;
    
    if ( this->_index_x != NULL) {
        delete this->_index_x;
        this->_index_x = NULL;
    }
    if ( this->_index_y != NULL) {
        delete this->_index_y;
        this->_index_y = NULL;
    }

}

void _cDatasetFlann::add_xy(double x[], double y[]) {

    for (int i = 0; i < this->dim_x; i++) {
        this->_data_x.push_back(x[i]);
    }
    for (int i = 0; i < this->dim_y; i++) {
        this->_data_y.push_back(y[i]);
    }
    
    // Creating the indexes
    if (this->size == 0) {

        flann::Matrix<double> M_data_x = flann::Matrix<double>(&this->_data_x[this->size], 1, this->dim_x);
        //this->_index_x = new flann::Index<L2<double> >(M_data_x, flann::AutotunedIndexParams(1.0, 0.01, 0.0, 0.1));
        //this->_index_x = new flann::Index<L2<double> >(M_data_x, flann::KmeansIndexParams(32, 11, FLANN_CENTERS_KMEANSPP, 0.2));
        this->_index_x = new flann::Index<L2<double> >(M_data_x, flann::LinearIndexParams(), L2<double>());
        this->_index_x->buildIndex();
        
        flann::Matrix<double> M_data_y = flann::Matrix<double>(&this->_data_y[this->size], 1, this->dim_y);
        //this->_index_y = new flann::Index<L2<double> >(M_data_y, flann::AutotunedIndexParams(1.0, 0.01, 0.0, 0.1));
        //this->_index_y = new flann::Index<L2<double> >(M_data_y, flann::KMeansIndexParams(32, 11, FLANN_CENTERS_KMEANSPP, 0.2));
        this->_index_y = new flann::Index<L2<double> >(M_data_y, flann::LinearIndexParams(), L2<double>());
        this->_index_y->buildIndex();

    } else { // or adding incrementaly

        const flann::Matrix<double> M_data_x = flann::Matrix<double>(&this->_data_x[this->size], 1, this->dim_x);
        this->_index_x->addPoints(M_data_x);
        this->_index_x->buildIndex();

        const flann::Matrix<double> M_data_y = flann::Matrix<double>(&this->_data_y[this->size], 1, this->dim_y);
        this->_index_y->addPoints(M_data_y);
        this->_index_y->buildIndex();
    }

    this->size += 1;
}

void _cDatasetFlann::get_x (int index, double x[]) {
    int offset = index*this->dim_x;
    for (int i = 0; i < this->dim_x; i++) {
        x[i] = this->_data_x[offset+i];
    }
}

void _cDatasetFlann::get_x_padded (int index, double x[]) {
    int offset = index*this->dim_x;
    x[0] = 1.0;
    for (int i = 0; i < this->dim_x; i++) {
        x[i+1] = this->_data_x[offset + i];
    }
}

void _cDatasetFlann::get_y (int index, double y[]) {
    int offset = index*this->dim_y;
    for (int i = 0; i < this->dim_y; i++) {
        y[i] = this->_data_y[offset+i];
    }
}

void _cDatasetFlann::nn_x(int knn, double xq[], double dists[], int index[]) {
    assert(knn < this->size);
    
    this->_nn_x(knn, xq);
    
    assert((int(this->_dists[0].size()) == knn) && (int(this->_index[0].size()) == knn));

    for (int i = 0; i < knn; i++) {
        dists[i] = this->_dists[0][i];
        index[i] = this->_index[0][i];
    }
}

void _cDatasetFlann::nn_y(int knn, double yq[], double dists[], int index[]) {
    assert(knn < this->size);
    
    this->_nn_y(knn, yq);
    
    assert(int(this->_dists[0].size()) == knn && int(this->_index[0].size()) == knn);
    
    for (int i = 0; i < knn; i++) {
        dists[i] = this->_dists[0][i];
        index[i] = this->_index[0][i];
    }
}

void _cDatasetFlann::nn_x(int knn, double xq[], vector<double>& dists, vector<int>& index) { 
    assert(knn < this->size);

    this->_nn_x(knn, xq);

    assert(int(this->_dists[0].size()) == knn && int(this->_index[0].size()) == knn);

    dists = this->_dists[0];
    index = this->_index[0];

    assert(int(dists.size()) == knn && int(index.size()) == knn);
}

void _cDatasetFlann::nn_y(int knn, double yq[], vector<double>& dists, vector<int>& index) { 
    assert(knn < this->size);
    
    this->_nn_y(knn, yq);
    
    assert(int(this->_dists[0].size()) == knn && int(this->_index[0].size()) == knn);
        
    dists = this->_dists[0];
    index = this->_index[0];
    
    assert(int(dists.size()) == knn && int(index.size()) == knn);
}

void _cDatasetFlann::_nn_x(int knn, double xq[]) {

    const flann::Matrix<double> Mxq = flann::Matrix<double>(xq, 1, this->dim_x);
    this->_index_x->knnSearch(Mxq, this->_index, this->_dists, knn, flann::SearchParams(32, 0.0, true));
}

void _cDatasetFlann::_nn_y(int knn, double yq[]) {

    const flann::Matrix<double> Myq = flann::Matrix<double>(yq, 1, this->dim_y);
    this->_index_y->knnSearch(Myq, this->_index, this->_dists, knn, flann::SearchParams(32, 0.0, true));
}
