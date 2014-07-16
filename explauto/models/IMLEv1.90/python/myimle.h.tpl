#ifndef MY_IMLE_H
#define MY_IMLE_H

#include <boost/python.hpp>
#include <string>

#include "imle.hpp"
#include "expert_public.hpp"

#define d $d
#define D $D
typedef IMLE<d, D, ::FastLinearExpert_public> IMLE_;


class ImleParam {
public:
    ImleParam();

    void set_param(//bool accelerated,
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
                   int iterMax) ;

    IMLE_::Param param;
};

class MyImle {
public:
    MyImle(const ImleParam &param);

    void update(const boost::python::list &z, const boost::python::list &x);

    boost::python::list predict(const boost::python::list &z);
    boost::python::list predictInverse(const boost::python::list &x);

    boost::python::list getJointMu(int expert);

    boost::python::list getInvSigma(int expert);
    boost::python::list getLambda(int expert);
    boost::python::list getPsi(int expert);

    int getNumberOfExperts();

    boost::python::list getPredictionWeight();
    boost::python::list getInversePredictionWeight();
    boost::python::list getPredictionVar();
    boost::python::list getInversePredictionVar();
    boost::python::list getPredictionJacobian();

    // boost::python::list getPsi0();
    // double getwPsi();

    std::string display();

private:
    IMLE_ _imle;
};

#endif
