#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

extern Eigen::VectorXd workNat(const Eigen::Map<Eigen::VectorXd> &para_tilde,
                               const bool &LEVIER, const int &Model_type,
                               const py::object &fixed_pars,
                               const py::object &fixed_values);

extern Eigen::VectorXd natWork(const Eigen::Map<Eigen::VectorXd> &para,
                               const bool &LEVIER, const int &Model_type);

extern Eigen::VectorXd volatilityVector(const Eigen::VectorXd &para, const int &K, const int &N);

extern Eigen::VectorXd probapi(const double &omega, const int &K, const int &N);

extern Eigen::MatrixXd P(const Eigen::VectorXd &para, const int &K, const int &N);

extern double logLik(const Eigen::Map<Eigen::VectorXd> &para_tilde,
                     const Eigen::MatrixXd &ech,
                     const int &Model_type,
                     const bool &LEVIER,
                     const int &K,
                     const int &N,
                     const int &Nl,
                     const py::object &fixed_pars,
                     const py::object &fixed_values,
                     const std::string &dis);

extern py::dict logLik2(const Eigen::MatrixXd &ech,
                        const Eigen::VectorXd &para,
                        const int &Model_type,
                        const bool &LEVIER,
                        const int &K,
                        const int &N,
                        const double &r,
                        const int &t,
                        const int &Nl,
                        const std::string &dis);

extern py::dict levierVolatility(const Eigen::VectorXd &ech,
                                 const Eigen::VectorXd &para,
                                 const int &Nl,
                                 const int &Model_type);

extern py::dict R_hat(const int &H,
                      const Eigen::Map<Eigen::VectorXd> &ech,
                      const Eigen::Map<Eigen::MatrixXi> &MC_sim,
                      const Eigen::Map<Eigen::MatrixXd> &z_t,
                      Eigen::MatrixXd Levier,
                      const Eigen::VectorXd &sig,
                      const Eigen::VectorXd &para,
                      const int &Model_type,
                      const int &N,
                      const int &Nl);

extern py::dict f_sim(const int &H,
                      const Eigen::Map<Eigen::VectorXd> &sig,
                      const Eigen::Map<Eigen::VectorXd> &pi_0,
                      const Eigen::Map<Eigen::MatrixXd> &matP,
                      const double &varphi,
                      const double &xi,
                      const double &shape,
                      const double &delta1,
                      const double &delta2);

PYBIND11_MODULE(mdsv_cpp, m) {
    m.doc() = "MDSV C++ implementation";

    m.def("workNat", &workNat, "Convert natural parameters to working parameters",
          py::arg("para_tilde"),
          py::arg("LEVIER") = false,
          py::arg("Model_type") = 0,
          py::arg("fixed_pars") = py::none(),
          py::arg("fixed_values") = py::none());

    m.def("natWork", &natWork, "Convert working parameters to natural parameters",
          py::arg("para"),
          py::arg("LEVIER") = false,
          py::arg("Model_type") = 0);

    m.def("volatilityVector", &volatilityVector, "Calculate volatility vector",
          py::arg("para"),
          py::arg("K"),
          py::arg("N"));

    m.def("probapi", &probapi, "Calculate stationary probabilities",
          py::arg("omega"),
          py::arg("K"),
          py::arg("N"));

    m.def("P", &P, "Calculate transition matrix",
          py::arg("para"),
          py::arg("K"),
          py::arg("N"));

    m.def("logLik", &logLik, "Calculate log-likelihood",
          py::arg("para_tilde"),
          py::arg("ech"),
          py::arg("Model_type") = 0,
          py::arg("LEVIER") = false,
          py::arg("K") = 2,
          py::arg("N") = 2,
          py::arg("Nl") = 70,
          py::arg("fixed_pars") = py::none(),
          py::arg("fixed_values") = py::none(),
          py::arg("dis") = "gamma");

    m.def("logLik2", &logLik2, "Calculate log-likelihood with additional outputs",
          py::arg("ech"),
          py::arg("para"),
          py::arg("Model_type") = 0,
          py::arg("LEVIER") = false,
          py::arg("K") = 2,
          py::arg("N") = 2,
          py::arg("r") = 0,
          py::arg("t") = 2,
          py::arg("Nl") = 70,
          py::arg("dis") = "gamma");

    m.def("levierVolatility", &levierVolatility, "Calculate leverage volatility",
          py::arg("ech"),
          py::arg("para"),
          py::arg("Nl") = 70,
          py::arg("Model_type") = 0);

    m.def("R_hat", &R_hat, "R_hat calculation",
          py::arg("H"),
          py::arg("ech"),
          py::arg("MC_sim"),
          py::arg("z_t"),
          py::arg("Levier"),
          py::arg("sig"),
          py::arg("para"),
          py::arg("Model_type") = 0,
          py::arg("N") = 3,
          py::arg("Nl") = 70);

    m.def("f_sim", &f_sim, "Forward simulation",
          py::arg("H"),
          py::arg("sig"),
          py::arg("pi_0"),
          py::arg("matP"),
          py::arg("varphi") = 0,
          py::arg("xi") = 0,
          py::arg("shape") = 0,
          py::arg("delta1") = 0,
          py::arg("delta2") = 0);
}