// The code works for Matrix of any dimension.

#include <iostream>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/SVD>

#include "predict.h"

using namespace Eigen;
using namespace std;

typedef double dtype;

typedef Matrix<dtype, 1, Dynamic, RowMajor>  VectorN;  // w
typedef DiagonalMatrix<dtype, Dynamic>         DiagN;  // W

// X and Y  (X have one more dimension than the useful data (filled with 1.0)).
typedef Matrix<dtype, 1, Dynamic, RowMajor>         VectorM;  // Xq
typedef Matrix<dtype, Dynamic, 1, 0>                ColumnM;  // Yq, pinv internals
typedef Matrix<dtype, Dynamic, Dynamic, RowMajor> MatrixNxM;  // Y, X, WX = W*X
typedef Matrix<dtype, Dynamic, Dynamic, RowMajor> MatrixMxN;  // WXT = WX.T, B = ((WXT*WX)ˆ-1)*WXT
typedef Matrix<dtype, Dynamic, Dynamic, RowMajor> MatrixMxM;  //  WXT*WX

void pseudoInverse(int dimX, MatrixMxM &a, MatrixMxM &result, double epsilon)
{
  JacobiSVD< MatrixMxM > svd = a.jacobiSvd(ComputeFullU | ComputeFullV);

  double tolerance = epsilon * std::max(a.cols(), a.rows()) * svd.singularValues().array().abs().maxCoeff();

  ColumnM single(dimX, 1);
  single = ((svd.singularValues().array().abs() > tolerance).select(svd.singularValues().array().inverse(), 0));

  result = svd.matrixV() 
          * single.asDiagonal()
          * svd.matrixU().adjoint();
}

void compute(int& dimX, int& dimY, int& dimNN, Map<VectorM>& Xq, Map<MatrixNxM>& X, Map<MatrixNxM>& Y, Map<VectorN>& w_nn, Map<ColumnM>& y) 
{    
    // cout << "Xq " << endl << Xq << endl;
    // cout << "X " << endl << X << endl;
    // cout << "Y " << endl << Y << endl;    
    DiagN W(dimNN);                 W = w_nn.asDiagonal();
    MatrixNxM WX(dimNN, dimX);      WX = W*X;
    MatrixMxN WXT(dimX, dimNN);    WXT = WX.transpose();
    // cout << WXT << endl;
    
    MatrixMxM WXTWX(dimX, dimX); WXTWX = WXT*WX;
    MatrixMxM WXTWX_inv(dimX, dimX);
    pseudoInverse(dimX, WXTWX, WXTWX_inv, 1e-15);  
    MatrixMxN B(dimX, dimNN);        B = WXTWX_inv*WXT;
    
    
                           y = Xq*B*W*Y;
}

void predictLWR(int dimX, int dimY, int dimNN, double Xq[], double X[], double Y[], double w[], double Yq[]) 
{
      Map<VectorM>   mXq(Xq, 1, dimX);
      Map<MatrixNxM> mX(X, dimNN, dimX);
      Map<MatrixNxM> mY(Y, dimNN, dimY);
      Map<VectorN>   mw(w, dimNN);
      Map<ColumnM>   mYq(Yq, dimY, 1);
  
      compute(dimX, dimY, dimNN, mXq, mX, mY, mw, mYq);

      // printf("X \n");
      // for (int i = 0; i < dimNN; i++) {
      //     for (int j = 0; j < dimX; j++) {
      //         printf("%6.2f ", X[dimX*i + j]);
      //     }
      //     printf("\n");      
      // }
      // printf("\n");      
      // printf("Y \n");
      // for (int i = 0; i < dimNN; i++) {
      //     for (int j = 0; j < dimY; j++) {
      //         printf("%6.2f ", Y[dimY*i + j]);
      //     }
      //     printf("\n");      
      // }
      // printf("\n");      
      // printf("Xq ");
      // for (int i = 0; i < dimX; i++) {
      //     printf("%6.2f ", Xq[i]);
      // }
      // printf("\n");
      // printf("Yq ");
      // for (int i = 0; i < dimY; i++) {
      //     printf("%6.2f ", Yq[i]);
      // }
      // printf("\n");
      // printf("w ");
      // for (int i = 0; i < dimNN; i++) {
      //     printf("%g ", w[i]);
      // }
      // printf("\n");      
}
