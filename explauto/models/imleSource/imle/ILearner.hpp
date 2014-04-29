#ifndef __ILEARNER_H
#define __ILEARNER_H

#include "EigenSerialized.h"

#include <string>


template< typename Input, typename Output >
class IOfflineLearner
{
public:
    virtual void addSamples(Input const &x, Output const &y) = 0;
    virtual void train() = 0;
    virtual Output const &predict( Input const &x ) = 0;
};


template< typename Input, typename Output >
class IOnlineLearner
{
public:
    virtual void update(Input const &x, Output const &y) = 0;
    virtual Output const &predict( Input const &x ) = 0;
};


template< int d, int D >
class NonLinearRegressor : IOnlineLearner< typename Eig<d,D>::Z, typename Eig<d,D>::X >
{
public:
    typedef typename Eig<d,D>::Z Z;
    typedef typename Eig<d,D>::X X;
    typedef typename Eig<d,D>::ZZ ZZ;
    typedef typename Eig<d,D>::XZ XZ;
    typedef typename Eig<d,D>::XX XX;

    typedef typename Eig<d,D>::ArrayZ ArrayZ;
    typedef typename Eig<d,D>::ArrayX ArrayX;
    typedef typename Eig<d,D>::ArrayZZ ArrayZZ;
    typedef typename Eig<d,D>::ArrayXX ArrayXX;

    // Identification
    virtual std::string getName() = 0;

    // Update
    virtual void update(Z const &x, X const &y) = 0;
    virtual void reset() = 0;

	virtual X const &predict(Z const &z) = 0;               	// Single Forward Prediction
	virtual void predictStrongest(Z const &z) = 0;          // Strongest Forward Prediction
	virtual void predictMultiple(Z const &z) = 0;  	            // Multi Forward Prediction

//	virtual void predictInverseSingle(X const &x) = 0;	    // Single Inverse Prediction
//	virtual void predictInverseStrongest(X const &x) = 0;	// Strongest Inverse Prediction
	virtual void predictInverse(X const &x) = 0;	            // Multiple Inverse Prediction

	virtual  ArrayVec  const &getPrediction() = 0;
	virtual  ArrayMat  const &getPredictionVar() = 0;
	virtual  ArrayScal const &getPredictionWeight() = 0;
	virtual  ArrayMat  const &getPredictionJacobian() = 0;
    virtual  int getNumberOfSolutionsFound() = 0;
	virtual Z const &getSigma() = 0;
	virtual X const &getPsi() = 0;
};


template< int d, int D >
class LinearModel
{
public:
    typedef typename Eig<d,D>::Z Z;
    typedef typename Eig<d,D>::X X;
    typedef typename Eig<d,D>::ZZ ZZ;
    typedef typename Eig<d,D>::XZ XZ;
    typedef typename Eig<d,D>::XX XX;

    Z Nu;
    X Mu;
    XZ Lambda;
    ZZ invSigma;
    X Psi;
};

template< int d, int D >
class MixtureOfLinearModels : public NonLinearRegressor< d, D >
{
public:
    typedef std::vector< LinearModel<d,D>, Eigen::aligned_allocator< LinearModel<d,D> > > LinearModels;

    virtual LinearModels getLinearModels() = 0;
    virtual int getNumberOfModels() = 0;
};


#endif

