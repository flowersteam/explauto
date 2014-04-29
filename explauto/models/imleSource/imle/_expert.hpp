//#include "expert.hpp"

#include <Eigen/LU>
#include <iostream>
#include <boost/math/special_functions/gamma.hpp>

#define EXPAND(var) #var "= " << var
#define EXPAND_N(var) #var "=\n" << var

#define EPSILON 0.000001


/*
 * LinearExpert Constructors
 */

template< int d, int D>
LinearExpert<d,D>::LinearExpert(Z const &z, X const &x, imle<d,D,::LinearExpert> *_mixture)
{
    mixture = _mixture;

    alpha = mixture->getParameters().alpha;
    wPsi = mixture->getParameters().wPsi;
    wNu = mixture->getParameters().wNu;
    wSigma = mixture->getParameters().wSigma;
    wLambda = mixture->getParameters().wLambda;

    Nu0 = z;


    H = EPSILON;

    Sh = 0.0;
    Sz.setZero();
    Sx.setZero();
    Sxz.setZero();
    Sxx.setZero();
    Szz.setIdentity() *= EPSILON;


    Nu = z;
    Mu = x;
    Lambda.setZero();
    varLambda = 1.0/wLambda * ZZ::Identity();

    Z sigma0 = wSigma/(wSigma+d+2.0)*mixture->getSigma();
    Sigma = sigma0.asDiagonal();
    invSigma = sigma0.cwiseInverse().asDiagonal();
    sqrtDetInvSigma = sqrt( 1.0 / sigma0.prod() );
//    Sigma.setIdentity() *=  wSigma*mixture->sigma/(wSigma+d+2.0);
//    invSigma.setIdentity() *=  (wSigma+d+2.0)/(wSigma*mixture->sigma);
//    sqrtDetInvSigma = pow((wSigma+d+2.0)/(wSigma*mixture->sigma), d/2.0);
    Psi = mixture->getPsi() / (1.0 + 2.0/wPsi) ;
    invPsi = Psi.cwiseInverse();

    recompute = true;
}

template< int d, int D>
LinearExpert<d,D>::LinearExpert()
{
    recompute = true;
}

// This is due to a STL annoyance (see http://blog.copton.net/archives/2007/10/13/stdvector/index.html)
template< int d, int D>
LinearExpert<d,D> & LinearExpert<d,D>::operator=(LinearExpert<d,D> const &other)
{
    std::cerr << "LinearExpert<d,D>::operator=(LinearExpert<d,D> const &other) --> YOU SHOULDN'T BE SEEING THIS!!\n";
    abort();

    return *this;
}

/*
 * LinearExpert Algorithm
 */
template< int d, int D>
void LinearExpert<d,D>::e_step( Z const &z, X const &x, Scal h )
{
    Scal decay = (pow(H+h,alpha) - h) / pow(H,alpha);

    H += h;

    Z zh = z * h;
	X xh = x * h;

	Sh  *= decay; Sh  += h;
	Sz  *= decay; Sz  += zh;
	Sx  *= decay; Sx  += xh;
	Sxz *= decay; Sxz.noalias() += xh * z.transpose();
	Szz *= decay; Szz.noalias() += zh * z.transpose();
	Sxx *= decay; Sxx += xh.cwiseProduct(x);

    wPsi *= decay;
    wNu *= decay;
    wSigma *= decay;
    wLambda *= decay;
}

template< int d, int D>
void LinearExpert<d,D>::m_step()
{
    Z meanZ = Sz / Sh;
    X meanX = Sx / Sh;

    if( wNu == 0.0 )
    {
        Nu = meanZ;
        Sigma = Szz;
//        Sigma += ( Z::Constant(wSigma*mixture->sigma) ).asDiagonal();
        Sigma += (wSigma*mixture->getSigma()).asDiagonal();
        Sigma.noalias() -= Sz*Nu.transpose();
        Sigma /= (Sh + wSigma + d + 1.0);
    }
    else
    {
        Nu = (wNu * Nu0 + Sz) / (wNu + Sh);
        Sigma = Szz;
//        Sigma += ( Z::Constant(wSigma*mixture->sigma) ).asDiagonal();
        Sigma += (wSigma*mixture->getSigma()).asDiagonal();
        Sigma.noalias() += (wNu*Nu0)*Nu0.transpose();
        Sigma.noalias() -= ((wNu+Sh)*Nu)*Nu.transpose();
        Sigma /= (Sh + wSigma + d + 2.0);
    }
	invSigma = Sigma.inverse();
    sqrtDetInvSigma = sqrt(invSigma.determinant());

    ZZ zz = Szz;
    zz += (wLambda * Z::Ones()).asDiagonal();
    zz.noalias() -= meanZ*Sz.transpose();
    varLambda = zz.inverse();
    XZ xz = Sxz;
    xz.noalias() -= meanX*Sz.transpose();
    Lambda.noalias() = xz * varLambda;
    Mu = meanX;
    Mu.noalias() += Lambda * (Nu - meanZ);

    if( 1.0/wPsi != 0.0 )
    {
        Psi = wPsi * mixture->getPsi();
        Psi.noalias() += Sxx - meanX.cwiseProduct(Sx) - xz.cwiseProduct(Lambda).rowwise().sum();
        Psi /= (wPsi + Sh + 2.0 );
        invPsi = Psi.cwiseInverse();
    }

    recompute = true;
}

template< int d, int D>
Scal LinearExpert<d,D>::queryH( Z const &z, X const &x )
{
    Z dz = z - Nu;
    X dx = x - (Lambda * dz + Mu);

    h = exp(-0.5*dz.dot(invSigma * dz)) * sqrtDetInvSigma * exp(-0.5*dx.cwiseAbs2().dot(invPsi)) * sqrt(invPsi.prod());

    return h;
}

template< int d, int D>
Scal LinearExpert<d,D>::queryZ( Z const &z )
{
    Z dz = z - Nu;
    pred_x = Lambda * dz + Mu;

    Scal dof = wSigma + Sh - d + 1.0;
    Scal aux1 = (wNu + Sh + 1.0) / (wNu + Sh) * (wSigma + Sh + d + (wNu == 0.0 ? 1.0 : 2.0));

    rbf_z = pow(1.0 + dz.dot( invSigma * dz)/aux1 , -0.5*(dof+d) );
    p_z_T = sqrtDetInvSigma / boost::math::tgamma_delta_ratio(0.5*dof, 0.5*d) * pow(2.0/aux1, 0.5*d) * rbf_z;

    dz = z - Sz/Sh;
    pred_x_var_factor = 1.0/Sh + dz.dot( varLambda * dz );

    return p_z_T;
}

template< int d, int D>
Scal LinearExpert<d,D>::queryX( X const &x )
{
    if( recompute )
    {
        PsiLambda = invPsi.asDiagonal() * Lambda;
        LambdaPsiLambda = Lambda.transpose() * PsiLambda;

        // Observation Variance
        pred_z_invVar = invSigma + LambdaPsiLambda;
        pred_z_var = pred_z_invVar.inverse();                      //TODO: Optimize this!

        p_x_invVar = - PsiLambda * pred_z_var * PsiLambda.transpose();
        p_x_invVar += invPsi.asDiagonal();
        p_x_invVarSqrtDet = sqrt(p_x_invVar.determinant());        //TODO: Optimize this!

        // Prediction Variance
//        pred_z_var_factor = pred_z_var * LambdaPsiLambda * pred_z_var;
//        if( wNu == 0.0 )
//            pred_z_var *= 1.0/Sh;
//        else
//        {
//            Z dz = Nu - Sz/Sh;
//            pred_z_var *= 1.0/Sh + dz.dot( varLambda * dz );
//        }

        recompute = false;
    }

    X dx = x - Mu;

    pred_z = Nu + pred_z_var * PsiLambda.transpose() * dx;
    p_x_Norm = p_x_invVarSqrtDet * exp( -0.5 * dx.dot(p_x_invVar * dx) );

    return p_x_Norm;
}

template< int d, int D>
Scal LinearExpert<d,D>::queryZX( Z const &z, X const &x )
{
    queryZ(z);

    X dx = x - pred_x;

    X xInvVar = invPsi / (1.0 + pred_x_var_factor);

    Scal rbf_x = exp(-0.5*dx.cwiseAbs2().dot(xInvVar));
    rbf_zx = rbf_z * rbf_x;
    p_zx = p_z_T * rbf_x * sqrt(xInvVar.prod());

    return p_zx;
}

/*
 * LinearExpert Display
 */
template< int d, int D>
void LinearExpert<d,D>::modelDisplay(std::ostream &out) const
{
    out << EXPAND_N(Nu) << std::endl;
    out << EXPAND_N(invSigma) << std::endl;
    out << EXPAND_N(Mu) << std::endl;
    out << EXPAND_N(Lambda) << std::endl;
    out << EXPAND_N(Psi) << std::endl;

//    out << EXPAND_N(H) << std::endl;
//    out << EXPAND_N(Sh) << std::endl;
//    out << EXPAND_N(Sz) << std::endl;
//    out << EXPAND_N(Sx) << std::endl;
//    out << EXPAND_N(Sxz) << std::endl;
//    out << EXPAND_N(Szz) << std::endl;
//    out << EXPAND_N(Sxx) << std::endl;
//    out << EXPAND_N(Nu0) << std::endl;
//
//    out << EXPAND(wPsi) << std::endl;
//    out << EXPAND(wNu) << std::endl;
//    out << EXPAND(wSigma) << std::endl;
//    out << EXPAND(wLambda) << std::endl;
//    out << EXPAND(alpha) << std::endl;
//    out << EXPAND(mixture->getPsi()) << std::endl;
//    out << EXPAND(mixture->getSigma()) << std::endl;
}

/*
 * FastLinearExpert Constructors
 */
template< int d, int D>
FastLinearExpert<d,D>::FastLinearExpert(Z const &z, X const &x, imle<d,D,::FastLinearExpert> *mixture)
{
    alpha = mixture->getParameters().alpha;
    wPsi = mixture->getParameters().wPsi;
    wNu = mixture->getParameters().wNu;
    wSigma = mixture->getParameters().wSigma + d + ((wNu==0) ? 1.0 : 2.0);
    wLambda = mixture->getParameters().wLambda;

    Nu0 = z;


    H = EPSILON;

    Sh = 0.0;
    Sz.setZero();
    Sx.setZero();
    Sxz.setZero();
    Sxx = wPsi * mixture->getPsi();
    invSzz.setIdentity() /= wLambda;
    Z Sigma0 = mixture->getParameters().wSigma*mixture->getSigma();
    invSzz0 = Sigma0.cwiseInverse().asDiagonal();
	// Using Moore-Penrose Rank-1 update
	Z SZ = invSzz0 * Nu0;
	Scal DOT = Nu0.dot(SZ);
	Scal DEN = DOT + 1 / wNu;
	invSzz0 -= (SZ/DEN) * SZ.transpose();

	// Using Determinant Rank-1 update
	detInvSzz0 = 1.0 / Sigma0.prod();
	detInvSzz0 /= (DOT*wNu + 1.0);

    Nu = z;
    Mu = x;
    Lambda.setZero();
    invSigma = (Sigma0/wSigma).cwiseInverse().asDiagonal();
    sqrtDetInvSigma = sqrt( 1.0 / (Sigma0/wSigma).prod() );

    Psi = mixture->getPsi() / (1.0 + 2.0/wPsi) ;
    invPsi = Psi.cwiseInverse();
//    invSzz.setIdentity() /= EPSILON;
//    invSzz0.setIdentity() /= this->w0 * this->sigma;
//    detInvSzz0 = 1.0 / pow(this->w0 * this->sigma, d);
//
//    this->Nu = z;
//    this->Mu = x;
//    this->Lambda.setZero();
//    this->invSSEzz.setZero();
//
//    this->invSigma.setIdentity() *=  (this->w0+d+1.0)/(this->w0*this->sigma);
//    this->sqrtDetInvSigma = pow((this->w0+d+1.0)/(this->w0*this->sigma), d/2.0);
//
//    this->recompute = true;
    recompute = true;
}

template< int d, int D>
FastLinearExpert<d,D>::FastLinearExpert()
: LinearExpert<d,D>()
{
}

// This is due to a STL annoyance (see http://blog.copton.net/archives/2007/10/13/stdvector/index.html)
template< int d, int D>
FastLinearExpert<d,D> & FastLinearExpert<d,D>::operator=(FastLinearExpert<d,D> const &other)
{
    std::cerr << "FastLinearExpert<d,D>::operator=(FastLinearExpert<d,D> const &other) --> YOU SHOULDN'T BE SEEING THIS!!\n";
    abort();

    return *this;
}


 /*
  * FastLinearExpert Algorithm
  */
template< int d, int D>
void FastLinearExpert<d,D>::e_step( Z const &z, X const &x, Scal h )
{
	Z SZ;
	Scal DOT, DEN;

    Scal decay = (pow(H+h,alpha) - h) / pow(H,alpha);

    H += h;

    Z zh = z * h;
	X xh = x * h;

	Sh  *= decay; Sh  += h;
	Sz  *= decay; Sz  += zh;
	Sx  *= decay; Sx  += xh;
	Sxz *= decay; Sxz.noalias() += xh * z.transpose();
	Sxx *= decay; Sxx += xh.cwiseProduct(x);

	invSzz /= decay;
	// Using Moore-Penrose Rank-1 update
	SZ = invSzz * z;
	DOT = z.dot(SZ);
	DEN = DOT + 1 / h;
	invSzz -= (SZ/DEN) * SZ.transpose();

	invSzz0 /= decay;
	// Using Moore-Penrose Rank-1 update
	SZ = invSzz0 * z;
	DOT = z.dot(SZ);
	DEN = DOT + 1 / h;
	invSzz0 -= (SZ/DEN) * SZ.transpose();

	// Using Determinant Rank-1 update
	detInvSzz0 /= pow(decay,d);
	detInvSzz0 /= (DOT*h + 1.0);

    // Priors decay...
    wPsi *= decay;
    wNu *= decay;
    wSigma *= decay;
    wLambda *= decay;
}

template< int d, int D>
void FastLinearExpert<d,D>::m_step()
{
	Z SZ;
	Scal DEN;
	Scal ShSigma = Sh + wSigma, ShNu = Sh + wNu;
    Z meanZ = Sz / Sh;
    X meanX = Sx / Sh;

    if( wNu == 0.0 )
        Nu = meanZ;
    else
        Nu = (Sz + wNu*Nu0) / ShNu;

	// Using Moore-Penrose Rank-1 downdate
	SZ = invSzz0 * Nu;
	DEN = Nu.dot(SZ) - 1.0/ShNu;
	invSigma = (invSzz0 - (SZ/DEN) * SZ.transpose()) * ShSigma;
	// Using Determinant Rank-1 downdate
	sqrtDetInvSigma = sqrt( -detInvSzz0 * ( pow(ShSigma,d) / ShNu / DEN) );


	// Using Moore-Penrose Rank-1 update
	SZ = invSzz * Sz;
	DEN = Sz.dot(SZ) - Sh;
	varLambda = invSzz - (SZ/DEN) * SZ.transpose();

    XZ xz = Sxz;
    xz.noalias() -= meanX*Sz.transpose();
    Lambda.noalias() = xz * varLambda;
    Mu = meanX;
    Mu.noalias() += Lambda * (Nu - meanZ);

    if( 1.0/wPsi != 0.0 )
    {
        Psi = Sxx;
        Psi.noalias() -= meanX.cwiseProduct(Sx) + xz.cwiseProduct(Lambda).rowwise().sum();
        Psi /= (wPsi + Sh + 2.0 );
        invPsi = Psi.cwiseInverse();
    }

    recompute = true;
}

/*
 * FastLinearExpert Display
 */
template< int d, int D>
void FastLinearExpert<d,D>::modelDisplay(std::ostream &out) const
{
    out << EXPAND_N(Nu) << std::endl;
    out << EXPAND_N(invSigma) << std::endl;
    out << EXPAND_N(Mu) << std::endl;
    out << EXPAND_N(Lambda) << std::endl;
    out << EXPAND_N(Psi) << std::endl;

    out << EXPAND_N(H) << std::endl;
    out << EXPAND_N(Sh) << std::endl;
    out << EXPAND_N(Sz) << std::endl;
    out << EXPAND_N(Sx) << std::endl;
    out << EXPAND_N(Sxz) << std::endl;
    out << EXPAND_N(invSzz) << std::endl;
    out << EXPAND_N(invSzz0) << std::endl;
    out << EXPAND_N(detInvSzz0) << std::endl;
    out << EXPAND_N(Sxx) << std::endl;
    out << EXPAND_N(Nu0) << std::endl;
//
//    out << EXPAND(wPsi) << std::endl;
//    out << EXPAND(wNu) << std::endl;
//    out << EXPAND(wSigma) << std::endl;
//    out << EXPAND(wLambda) << std::endl;
//    out << EXPAND(alpha) << std::endl;
//    out << EXPAND(mixture->getPsi()) << std::endl;
//    out << EXPAND(mixture->getSigma()) << std::endl;
}



/*
 * LinearExpert Inline members
 */
template< int d, int D>
template<typename Archive>
void LinearExpert<d,D>::serialize(Archive & ar, const unsigned int version)
{
    ar & mixture;

    ar & H;
    ar & Sh;
    ar & Sz;
    ar & Sx;
    ar & Sxz;
    ar & Szz;
    ar & Sxx;

    ar & Nu0;

    ar & alpha;
    Scal wPsiInv = 1.0/wPsi;
    ar & wPsiInv;
    wPsi = 1.0/wPsiInv;
    ar & wNu;
    ar & wSigma;
    ar & wLambda;

    ar & Nu;
    ar & Mu;
    ar & Lambda;
    ar & invSigma;
    ar & Psi;
    ar & invPsi;
    ar & sqrtDetInvSigma;
    ar & varLambda;
}

template< int d, int D>
template<typename Archive>
void FastLinearExpert<d,D>::serialize(Archive & ar, const unsigned int version)
{
    ar & H;
    ar & Sh;
    ar & Sz;
    ar & Sx;
    ar & Sxz;
    ar & Sxx;
    ar & invSzz;
    ar & invSzz0;
    ar & detInvSzz0;

    ar & Nu0;

    ar & alpha;
    Scal wPsiInv = 1.0/wPsi;
    ar & wPsiInv;
    wPsi = 1.0/wPsiInv;
    ar & wNu;
    ar & wSigma;
    ar & wLambda;

    ar & Nu;
    ar & Mu;
    ar & Lambda;
    ar & invSigma;
    ar & Psi;
    ar & invPsi;
    ar & sqrtDetInvSigma;
    ar & varLambda;
}

template< int d, int D>
Scal LinearExpert<d,D>::get_h() const
{
    return h;
}

template< int d, int D>
Scal LinearExpert<d,D>::get_p_z() const
{
    return p_z_T;
}

template< int d, int D>
Scal LinearExpert<d,D>::get_p_x() const
{
    return p_x_Norm;
}

template< int d, int D>
Scal LinearExpert<d,D>::get_p_zx() const
{
    return p_zx;
}

template< int d, int D>
Scal LinearExpert<d,D>::get_rbf_zx() const
{
    return rbf_zx;
}


template< int d, int D>
typename LinearExpert<d,D>::X const &LinearExpert<d,D>::getPredX() const
{
    return pred_x;
}

template< int d, int D>
typename LinearExpert<d,D>::X const &LinearExpert<d,D>::getPredXVar() const
{
    return Psi;
}

template< int d, int D>
typename LinearExpert<d,D>::X const &LinearExpert<d,D>::getPredXInvVar() const
{
    return invPsi;
}

template< int d, int D>
Scal LinearExpert<d,D>::getUncertFactorPredX() const
{
    return pred_x_var_factor;
}


template< int d, int D>
typename LinearExpert<d,D>::Z const &LinearExpert<d,D>::getPredZ() const
{
    return pred_z;
}

template< int d, int D>
typename LinearExpert<d,D>::ZZ const &LinearExpert<d,D>::getPredZVar() const
{
    return pred_z_var;
}

template< int d, int D>
typename LinearExpert<d,D>::ZZ const &LinearExpert<d,D>::getPredZInvVar() const
{
    return pred_z_invVar;
}

template< int d, int D>
typename LinearExpert<d,D>::ZZ const & LinearExpert<d,D>::getUncertFactorPredZ() const
{
    return pred_z_var_factor;
}
