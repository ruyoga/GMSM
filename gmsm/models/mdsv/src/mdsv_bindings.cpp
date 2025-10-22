// mdsv_bindings.cpp - Pybind11 bindings for MDSV C++ core
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "mdsv_core.cpp"

namespace py = pybind11;

PYBIND11_MODULE(mdsv_cpp, m) {
    m.doc() = "MDSV C++ core implementation with leverage effect support";

    py::class_<MDSVCore>(m, "MDSVCore")
        .def(py::init<int, int>(),
             py::arg("K"),
             py::arg("N"),
             "Initialize MDSV core with K states per component and N components")

        .def("work_nat", &MDSVCore::workNat,
             py::arg("para_tilde"),
             py::arg("LEVIER") = false,
             py::arg("Model_type") = 0,
             "Transform working parameters to natural parameters")

        .def("nat_work", &MDSVCore::natWork,
             py::arg("para"),
             py::arg("LEVIER") = false,
             py::arg("Model_type") = 0,
             "Transform natural parameters to working parameters")

        .def("volatility_vector", &MDSVCore::volatilityVector,
             py::arg("para"),
             py::arg("K"),
             py::arg("N"),
             "Compute volatility vector")

        .def("transition_matrix", &MDSVCore::transitionMatrix,
             py::arg("para"),
             py::arg("K"),
             py::arg("N"),
             "Compute transition matrix")

        .def("log_likelihood", &MDSVCore::logLikelihood,
             py::arg("para_tilde"),
             py::arg("data"),
             py::arg("Model_type") = 0,
             py::arg("LEVIER") = false,
             "Compute log-likelihood WITH leverage effect support")

        .def("compute_leverage_multipliers", &MDSVCore::computeLeverageMultipliers,
             py::arg("returns"),
             py::arg("l1"),
             py::arg("theta_l"),
             py::arg("NL") = 70,
             "Compute leverage multipliers Lt from return series");

    // Module-level functions for compatibility
    m.def("work_nat", [](const Eigen::VectorXd& para_tilde, bool LEVIER, int Model_type) {
        MDSVCore core(2, 2);  // Dummy K, N values
        return core.workNat(para_tilde, LEVIER, Model_type);
    }, py::arg("para_tilde"), py::arg("LEVIER") = false, py::arg("Model_type") = 0,
       "Transform working parameters to natural parameters");

    m.def("nat_work", [](const Eigen::VectorXd& para, bool LEVIER, int Model_type) {
        MDSVCore core(2, 2);  // Dummy K, N values
        return core.natWork(para, LEVIER, Model_type);
    }, py::arg("para"), py::arg("LEVIER") = false, py::arg("Model_type") = 0,
       "Transform natural parameters to working parameters");

    m.def("volatility_vector", [](const Eigen::VectorXd& para, int K, int N) {
        MDSVCore core(K, N);
        return core.volatilityVector(para, K, N);
    }, py::arg("para"), py::arg("K"), py::arg("N"),
       "Compute volatility vector");

    m.def("transition_matrix", [](const Eigen::VectorXd& para, int K, int N) {
        MDSVCore core(K, N);
        return core.transitionMatrix(para, K, N);
    }, py::arg("para"), py::arg("K"), py::arg("N"),
       "Compute transition matrix");

    m.def("log_likelihood", [](const Eigen::VectorXd& para_tilde,
                               const Eigen::MatrixXd& data,
                               int K, int N,
                               int Model_type, bool LEVIER) {
        MDSVCore core(K, N);
        return core.logLikelihood(para_tilde, data, Model_type, LEVIER);
    }, py::arg("para_tilde"), py::arg("data"), py::arg("K"), py::arg("N"),
       py::arg("Model_type") = 0, py::arg("LEVIER") = false,
       "Compute log-likelihood with leverage effect");

    m.def("compute_leverage_multipliers", [](const Eigen::VectorXd& returns,
                                             double l1, double theta_l, int NL) {
        MDSVCore core(2, 2);  // Dummy K, N
        return core.computeLeverageMultipliers(returns, l1, theta_l, NL);
    }, py::arg("returns"), py::arg("l1"), py::arg("theta_l"), py::arg("NL") = 70,
       "Compute leverage multipliers");
}