#include "myimle.h"

ImleParam::ImleParam() {
}

void ImleParam::set_param(//bool accelerated,
                          double alpha,
                          const boost::python::list &Psi0,
                          double sigma0,
                          double wsigma,
                          double wpsi,
                          double wNu,
                          double wLambda,
                          double wSigma,
                          double wPsi,
                          int nOutliers,
                          double p0,
                          double multiValuedSignificance,
                          int nSolMin,
                          int nSolMax,
                          int iterMax) {

    //param.accelerated = accelerated;
    param.alpha = alpha;

    Eig<d, D>::X psi0;
    for (int i=0; i < boost::python::len(Psi0); i++) {
        psi0[i] = boost::python::extract<double>(Psi0[i]);
    }
    param.Psi0 = psi0;

    param.sigma0 = sigma0;
    param.wsigma = wsigma;
    param.wpsi = wpsi;
    param.wNu = wNu;
    param.wLambda = wLambda;
    param.wSigma = wSigma;
    param.wPsi = wPsi;
    param.nOutliers = nOutliers;
    param.p0 = p0;
    param.multiValuedSignificance = multiValuedSignificance;
    param.nSolMin = nSolMin;
    param.nSolMax = nSolMax;
    param.iterMax  = iterMax;


}

MyImle::MyImle(const ImleParam &param) {
    _imle = IMLE_(param.param);
}

void MyImle::update(const boost::python::list &z, const boost::python::list &x) {
    IMLE_::Z _z;
    IMLE_::X _x;

    for (int i=0; i < boost::python::len(x); i++) {
        _x[i] = boost::python::extract<double>(x[i]);
    }

    for (int i=0; i < boost::python::len(z); i++) {
        _z[i] = boost::python::extract<double>(z[i]);
    }

    _imle.update(_z, _x);
}

boost::python::list MyImle::predict(const boost::python::list &z) {
    IMLE_::Z _z;

    for (int i=0; i < boost::python::len(z); i++) {
        _z[i] = boost::python::extract<double>(z[i]);
    }

    IMLE_::X _x;
    _x = _imle.predict(_z);

    boost::python::list l;

    for (int i=0; i < _x.count(); i++) {
        l.append(_x[i]);
    }

    return l;
}

boost::python::list MyImle::predictInverse(const boost::python::list &x) {
    IMLE_::X _x;

    for (int i=0; i < boost::python::len(x); i++) {
        _x[i] = boost::python::extract<double>(x[i]);
    }

    _imle.predictInverse(_x);
    const IMLE_::ArrayZ inv_preds = _imle.getInversePredictions();
    boost::python::list invSols;
    int nSol = inv_preds.size();
    for( int k = 0; k < nSol; k++ ) {
        boost::python::list sol;
        for( int i = 0; i < inv_preds[0].size(); i++ ) {
            sol.append(inv_preds[k][i]);
        }
        invSols.append(sol);
    }

    return invSols;
}

boost::python::list MyImle::getJointMu(int expert) {
    boost::python::list l;
    for(int i=0; i<d; i++)
        l.append(_imle.getExperts()[expert].Nu[i]);

    for(int i=0; i<D; i++)
        l.append(_imle.getExperts()[expert].Mu[i]);

    return l;
}

boost::python::list MyImle::getInvSigma(int expert) {
    IMLE_::ZZ A=_imle.getExperts()[expert].getInvSigma();
    boost::python::list ll;
	for(int i=0; i<d; i++) {
	    boost::python::list l;
	    for(int j=0; j<d; j++)
            	l.append(A(i, j));
	    ll.append(l);
	}
    return ll;
}

boost::python::list MyImle::getLambda(int expert) {
    IMLE_::XZ A=_imle.getExperts()[expert].getLambda();
    boost::python::list ll;
	for(int i=0; i<D; i++) {
	    boost::python::list l;
	    for(int j=0; j<d; j++)
            	l.append(A(i, j));
	    ll.append(l);
	}
    return ll;
}

boost::python::list MyImle::getPsi(int expert) {
    IMLE_::X A=_imle.getExperts()[expert].getPsi();
    boost::python::list l;
	for(int i=0; i<D; i++) {
		l.append(A(i));
	}
    return l;
}

int MyImle::getNumberOfExperts() {
    return _imle.getExperts().size();
}

boost::python::list MyImle::getPredictionWeight() {
    boost::python::list l;
    const Scal weights = _imle.getPredictionWeight();
    l.append(weights);
    return l;
}

boost::python::list MyImle::getInversePredictionWeight() {
    boost::python::list l;
    const ArrayScal weights = _imle.getInversePredictionsWeight();
    for (int i=0; i < _imle.getNumberOfInverseSolutionsFound(); i++) {
        l.append(weights[i]);
    }
    return l;
}

boost::python::list MyImle::getPredictionVar() {
    IMLE_::X A = _imle.getPredictionVar();
    boost::python::list l;

    for (int k=0; k < A.size(); k++)
        l.append(A[k]);
    return l;
}

boost::python::list MyImle::getInversePredictionVar() {
    IMLE_::ArrayZZ A = _imle.getInversePredictionsVar();
    boost::python::list lll;

    for (int k=0; k < A.size(); k++) {
        boost::python::list ll;

        for(int i=0; i < A[k].rows(); i++) {
            boost::python::list l;
            for(int j=0; j < A[k].cols(); j++)
                l.append(A[k](i, j));
            ll.append(l);
        }
        lll.append(ll);
    }
    return lll;
}

boost::python::list MyImle::getPredictionJacobian() {
    IMLE_::XZ A = _imle.getPredictionJacobian();
    boost::python::list ll;

    for (int i=0; i < D; i++) {
        boost::python::list l;

        for(int j=0; j < d; j++)
            l.append(A(i, j));
        ll.append(l);
    }
    return ll;
}

// boost::python::list MyImle::getPsi0() {
//     Eig<d,D>::X psi0 = _imle.getParameters().Psi0;

//     boost::python::list l;

//     for (int i=0; i < psi0.size(); i++) {
//         l.append(psi0[i]);
//     }
//     return l;
// }

// double MyImle::getwPsi() {
//     return _imle.getParameters().wPsi;
// }

std::string MyImle::display() {
    std::ostringstream s;

    _imle.modelDisplay(s);

    return s.str();
}
