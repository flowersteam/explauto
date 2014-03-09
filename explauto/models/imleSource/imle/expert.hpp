#ifndef __EXPERT_H
#define __EXPERT_H

#include "EigenSerialized.h"
#include "ILearner.hpp"
//#include "../python/myimle.h"

/*
 * LinearExpert Interface
 */

template< int d, int D, template<int,int> class Expert>
class imle;

template< int d, int D>
class LinearExpert : public LinearModel<d,D>
{
public:
    typedef typename Eig<d,D>::Z Z;
    typedef typename Eig<d,D>::X X;
    typedef typename Eig<d,D>::ZZ ZZ;
    typedef typename Eig<d,D>::XZ XZ;
    typedef typename Eig<d,D>::XX XX;

    // Constructors
    LinearExpert(Z const &z, X const &x, imle<d,D,::LinearExpert> *mixture);
    LinearExpert();                                                 // Needed for Boost Serialization...
    LinearExpert<d,D> & operator=(LinearExpert<d,D> const &other);  // This is due to a STL annoyance (see http://blog.copton.net/archives/2007/10/13/stdvector/index.html)

    // Update
    void e_step( Z const &z, X const &x, Scal h );
    void m_step();

    // Predict
    Scal queryH( Z const &z, X const &x );
    Scal queryZ( Z const &z);
    Scal queryX( X const &x);
    Scal queryZX( Z const &z, X const &x );

    // Gets and Sets
    inline Scal get_h() const;
    inline Scal get_p_z() const;
    inline Scal get_p_x() const;
    inline Scal get_p_zx() const;
    inline Scal get_rbf_zx() const;

    inline X const &getPredX() const;
    inline X const &getPredXVar() const;
    inline X const &getPredXInvVar() const;
    inline Scal getUncertFactorPredX() const;

    inline Z const &getPredZ() const;
    inline ZZ const &getPredZVar() const;
    inline ZZ const &getPredZInvVar() const;
    inline ZZ const &getUncertFactorPredZ() const;

    // Display
	void modelDisplay(std::ostream &out = std::cout) const;

    using LinearModel<d,D>::Nu;
    using LinearModel<d,D>::Mu;
    using LinearModel<d,D>::Lambda;
    using LinearModel<d,D>::invSigma;
    using LinearModel<d,D>::Psi;
    X invPsi;

	//friend class MyImle;
protected:
    // Boost serialization
    friend class boost::serialization::access;
    template<typename Archive>
    inline void serialize(Archive & ar, const unsigned int version);

    // Shared prior
    imle<d,D,::LinearExpert> *mixture;

    // Memory Traces
    Scal H;        // Needed to decay statistics
    Scal Sh;
    Z Sz;
    X Sx;
    XZ Sxz;
    ZZ Szz;
    X Sxx;

    // Priors parameters and decay
    Z Nu0;
    Scal alpha;
    Scal wPsi;
    Scal wNu;
    Scal wSigma;
    Scal wLambda;

    // Recomputing
    bool recompute;

    // Storing results
    Scal h, p_z_T, p_x_Norm, p_zx, rbf_z, rbf_zx;

    X pred_x;
    Z pred_z;

    Scal pred_x_var_factor;
    ZZ pred_z_var_factor;
    ZZ pred_z_var;
    ZZ pred_z_invVar;

	// Aux variables
    Scal sqrtDetInvSigma;
    ZZ varLambda;

    ZZ Sigma;
    XZ PsiLambda;
    ZZ LambdaPsiLambda;
    XX p_x_invVar;
    Scal p_x_invVarSqrtDet;
};

/*
 * FastLinearExpert Interface
 */
template< int d, int D>
class FastLinearExpert : public LinearExpert<d,D>
{
public:
    // Inherited from base class LinearExpert<d,D>
    typedef typename LinearExpert<d,D>::Z Z;
    typedef typename LinearExpert<d,D>::X X;
    typedef typename LinearExpert<d,D>::ZZ ZZ;
    typedef typename LinearExpert<d,D>::XZ XZ;
    typedef typename LinearExpert<d,D>::XX XX;

    FastLinearExpert(Z const &z, X const &x, imle<d,D,::FastLinearExpert> *mixture);
    FastLinearExpert();
    FastLinearExpert<d,D> & operator=(FastLinearExpert<d,D> const &other);  // This is due to a STL annoyance (see http://blog.copton.net/archives/2007/10/13/stdvector/index.html)

    void e_step( Z const &z, X const &x, Scal h );
    void m_step();

    using LinearModel<d,D>::Nu;
    using LinearModel<d,D>::Mu;
    using LinearModel<d,D>::Lambda;
    using LinearModel<d,D>::invSigma;
    using LinearModel<d,D>::Psi;
    using LinearExpert<d,D>::invPsi;

    void modelDisplay(std::ostream &out = std::cout) const;

protected:
    // Boost serialization
    friend class boost::serialization::access;
    template<typename Archive>
    inline void serialize(Archive & ar, const unsigned int version);

    // Memory Traces
    using LinearExpert<d,D>::H;
    using LinearExpert<d,D>::Sh;
    using LinearExpert<d,D>::Sz;
    using LinearExpert<d,D>::Sx;
    using LinearExpert<d,D>::Sxz;
    using LinearExpert<d,D>::Sxx;
    ZZ invSzz;
    ZZ invSzz0;
    Scal detInvSzz0;

    // Priors parameters and decay
    using LinearExpert<d,D>::Nu0;
    using LinearExpert<d,D>::alpha;
    using LinearExpert<d,D>::wPsi;
    using LinearExpert<d,D>::wNu;
    using LinearExpert<d,D>::wSigma;
    using LinearExpert<d,D>::wLambda;

    // Recomputing
    using LinearExpert<d,D>::recompute;

	// Aux variables
    using LinearExpert<d,D>::sqrtDetInvSigma;
    using LinearExpert<d,D>::varLambda;
};

// Expert template implementation
#include "_expert.hpp"

#endif


