#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include "myimle.h"

using namespace boost::python;

BOOST_PYTHON_MODULE(_imle)
{
    class_<ImleParam>("ImleParam", init<>())
        .def("set_param", &ImleParam::set_param)
    ;

    class_<MyImle>("Imle", init<const ImleParam &>())
        .def("update", &MyImle::update)

        .def("predict", &MyImle::predict)
        .def("predict_inverse", &MyImle::predictInverse)

        .def("get_joint_mu", &MyImle::getJointMu)

	    .def("get_inv_sigma", &MyImle::getInvSigma)

	    .def("get_lambda", &MyImle::getLambda)

        .def("get_psi", &MyImle::getPsi)

        .def("get_number_of_experts", &MyImle::getNumberOfExperts)

        .def("getPredictionWeight", &MyImle::getPredictionWeight)

        // .def("get_psi0", &MyImle::getPsi0)
        // .def("get_wPsi", &MyImle::getwPsi)

        .def("display", &MyImle::display)
    ;
}
