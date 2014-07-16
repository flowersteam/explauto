#ifndef __EXPERT_PUBLIC_H
#define __EXPERT_PUBLIC_H

#include "expert.hpp"

template< int d, int D>
class FastLinearExpert_public : public FastLinearExpert<d,D>
{
public:
	typedef typename Eig<d,D>::Z Z;
	typedef typename Eig<d,D>::X X;
	typedef typename Eig<d,D>::ZZ ZZ;
	typedef typename Eig<d,D>::XZ XZ;
	typedef typename Eig<d,D>::XX XX;
	ZZ getInvSigma() const {
		return this->invSigma;
	}
	XZ getLambda() const {
		return this->Lambda;
	}
	X getPsi() const {
		return this->Psi;
	}
	//using FastLinearExpert<d,D>::FastLinearExpert;
	FastLinearExpert_public(Z const &z, X const &x, IMLE<d,D,::FastLinearExpert_public > *mixture) : FastLinearExpert<d,D>(z, x, (IMLE<d,D,::FastLinearExpert> *)mixture){
		;
	}

};

#endif
