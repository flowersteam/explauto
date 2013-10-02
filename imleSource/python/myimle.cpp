#include "myimle.h"

ImleParam::ImleParam() {
}

void ImleParam::set_param(double alpha,
                           const boost::python::list &Psi0,
                           double sigma0,
                           double wsigma,
                           double wSigma,
                           double wNu,
                           double wLambda,
                           double wPsi,
                           double p0,
                           double multiValuedSignificance,
                           int nSolMax) {

    param.alpha = alpha;

    Eig<d, D>::X psi0;
    for (int i=0; i < boost::python::len(Psi0); i++) {
        psi0[i] = boost::python::extract<double>(Psi0[i]);
    }
    param.Psi0 = psi0;

    param.sigma0 = sigma0;
    param.wsigma = wsigma;
    param.wSigma = wSigma;
    param.wNu = wNu;
    param.wLambda = wLambda;
    param.wPsi = wPsi;
    param.p0 = p0;
    param.multiValuedSignificance = multiValuedSignificance;
    param.nSolMax = nSolMax;
}

MyImle::MyImle(const ImleParam &param) {
    _imle = IMLE(param.param);
}

void MyImle::update(const boost::python::list &z, const boost::python::list &x) {
    IMLE::Z _z;
    IMLE::X _x;

    for (int i=0; i < boost::python::len(x); i++) {
        _x[i] = boost::python::extract<double>(x[i]);
    }

    for (int i=0; i < boost::python::len(z); i++) {
        _z[i] = boost::python::extract<double>(z[i]);
    }

    _imle.update(_z, _x);
}

boost::python::list MyImle::predict(const boost::python::list &z) {
    IMLE::Z _z;

    for (int i=0; i < boost::python::len(z); i++) {
        _z[i] = boost::python::extract<double>(z[i]);
    }

    IMLE::X _x;
    _x = _imle.predict(_z);

    boost::python::list l;

    for (int i=0; i < _x.count(); i++) {
        l.append(_x[i]);
    }

    return l;
}

boost::python::list MyImle::predictInverse(const boost::python::list &x) {
    IMLE::X _x;

    for (int i=0; i < boost::python::len(x); i++) {
        _x[i] = boost::python::extract<double>(x[i]);
    }

    _imle.predictInverse(_x);


    boost::python::list l;

    const ArrayVec &pred = _imle.getPrediction();
    for (int i=0; i < pred.size(); i++) {
        const Vec &v = pred[i];

        boost::python::list l1;

        for (int j=0; j < v.size(); j++) {
            l1.append(v[j]);
        }

        l.append(l1);
    }

    return l;
}

boost::python::list MyImle::getJointMu(int expert) {
    boost::python::list l;
    for(int i=0; i<d; i++)
        l.append(_imle.getExperts()[expert].Nu[i]);

    for(int i=0; i<D; i++)
        l.append(_imle.getExperts()[expert].Mu[i]);

    return l;
}

boost::python::list MyImle::getJointSigma(int expert) {
    IMLE::ZZ A=_imle.getExperts()[expert].Sigma;
    boost::python::list l;
    for(int i=0; i<d; i++)
        l.append(_imle.getExperts()[expert].Nu[i]);

    for(int i=0; i<D; i++)
        l.append(_imle.getExperts()[expert].Mu[i]);

    return l;
}


int MyImle::getNumberOfExperts() {
    return _imle.getExperts().size();
}

boost::python::list MyImle::getPredictionWeight() {
    boost::python::list l;

    for (int i=0; i < _imle.getNumberOfSolutionsFound(); i++) {
        l.append(_imle.getPredictionWeight()[i]);
    }
    return l;
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

