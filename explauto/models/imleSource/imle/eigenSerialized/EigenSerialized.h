#ifndef __EIGENSERIALIZED_H
#define __EIGENSERIALIZED_H


// Boost
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>

// Note: BEFORE eigen includes!!
//#define EIGEN_MATRIXBASE_PLUGIN "EigenMatrixBaseSerialize.h"
#define EIGEN_PLAINOBJECTBASE_PLUGIN "EigenMatrixBaseSerialize.h"

// Eigen
#include <Eigen/Core>
#include<Eigen/StdVector>


typedef Eigen::VectorXd Vec;
typedef Eigen::MatrixXd Mat;
typedef double   Scal;
typedef std::vector<Vec> ArrayVec;
typedef std::vector<Mat> ArrayMat;
typedef std::vector<Scal> ArrayScal;


template<int d, int D>
class Eig
{
    public:
    typedef Eigen::Matrix<Scal, d, 1> Z;
    typedef Eigen::Matrix<Scal, D, 1> X;
    typedef Eigen::Matrix<Scal, d, d> ZZ;
    typedef Eigen::Matrix<Scal, D, d> XZ;
    typedef Eigen::Matrix<Scal, D, D> XX;
    typedef Eigen::DiagonalMatrix<Scal, d> diagZZ;
    typedef Eigen::DiagonalMatrix<Scal, D> diagXX;

    typedef typename std::vector<Z, Eigen::aligned_allocator<Z> > ArrayZ;
    typedef typename std::vector<X, Eigen::aligned_allocator<X> > ArrayX;
    typedef typename std::vector<ZZ, Eigen::aligned_allocator<ZZ> > ArrayZZ;
    typedef typename std::vector<XX, Eigen::aligned_allocator<XX> > ArrayXX;
    typedef typename std::vector<XZ, Eigen::aligned_allocator<XZ> > ArrayXZ;
};


#endif
