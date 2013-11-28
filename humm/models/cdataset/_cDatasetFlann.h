#ifndef _CDATASETFLANN_H_
#define _CDATASETFLANN_H_

#include <flann/flann.h>
#include <vector>

using namespace std;

class _cDatasetFlann {
    public:
        _cDatasetFlann(int dim_x, int dim_y);
        ~_cDatasetFlann();
        int dim_x;
        int dim_y;
        int size;

        void reset();

        void add_xy(double x[], double y[]);

        void get_x (int index, double x[]);
        void get_x_padded (int index, double x[]);
        void get_y (int index, double y[]);

        void nn_x(int knn, double xq[], double dists[], int index[]);
        void nn_y(int knn, double yq[], double dists[], int index[]);

        // Same with vector
        void nn_x(int knn, double xq[], vector<double>& dists, vector<int>& index);
        void nn_y(int knn, double yq[], vector<double>& dists, vector<int>& index);

        // Data (read only !)
        vector<double> _data_x;
        vector<double> _data_y;

    private:


        // KDtree
        flann::Index<L2<double> >* _index_x;
        flann::Index<L2<double> >* _index_y;

        // Temporary data
        vector<vector<int> >    _index;
        vector<vector<double> > _dists;

        void _nn_x(int k, double xq[]);
        void _nn_y(int k, double yq[]);

};

#endif // _CDATASETFLANN_H_
