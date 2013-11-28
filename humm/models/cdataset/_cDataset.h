#ifndef _CDATASET_H_
#define _CDATASET_H_

#include <vector>
#include <queue>

class _cDataset {
    public:
        _cDataset(int dim_x, int dim_y);
        ~_cDataset();
        int dim_x;
        int dim_y;
        int size;

        void reset();

        void add_xy(double x[], double y[]);
        void add_xy(std::vector<double>& x, std::vector<double>& y);

        void get_x (int index, double x[]);
        void get_y (int index, double y[]);
        void get_x_padded (int index, double x[]);

        void get_x (int index, std::vector<double>& x);
        void get_y (int index, std::vector<double>& y);
        void get_x_padded (int index, std::vector<double>& x);

        void nn_x(int knn, double xq[], double dists[], int index[]);
        void nn_y(int knn, double yq[], double dists[], int index[]);
        void nn_xy(int knn, double xq[], double yq[], double dists[], int index[], double w_x, double w_y);

        // Same with vector
        void nn_x_v(int knn, double xq[], std::vector<double>& dists, std::vector<int>& index);
        void nn_y_v(int knn, double yq[], std::vector<double>& dists, std::vector<int>& index);
        void nn_xy_v(int knn, double xq[], double yq[], std::vector<double>& dists, std::vector<int>& index, double w_x, double w_y);

        std::vector< std::vector<double> > _data_x;
        std::vector< std::vector<double> > _data_y;

    private:

        // Temporary data
        std::vector<int>    _index;
        std::vector<double> _dists;

        std::priority_queue<std::pair<double, int>, std::vector<std::pair<double, int> >, std::greater<std::pair<double, int> > > _heap;

        double L2(std::vector<double>& a, std::vector<double>&b);

};

#endif // _CDATASET_H_
