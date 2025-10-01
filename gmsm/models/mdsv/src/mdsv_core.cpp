// mdsv_core.cpp - Core C++ implementation for MDSV
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

using namespace Eigen;

class MDSVCore {
public:
    // Constructor
    MDSVCore(int K, int N) : K_(K), N_(N) {
        state_size_ = std::pow(K, N);
    }

    // Work/Natural parameter transformation
    VectorXd workNat(const VectorXd& para_tilde, bool LEVIER = false,
                     int Model_type = 0) {
        VectorXd para = para_tilde;

        para(0) = 1.0 / (1.0 + std::exp(para_tilde(0)));  // omega
        para(1) = 1.0 / (1.0 + std::exp(para_tilde(1)));  // a
        para(2) = 1.0 + std::exp(para_tilde(2));          // b
        para(3) = std::exp(para_tilde(3));                // sigma
        para(4) = 1.0 / (1.0 + std::exp(para_tilde(4)));  // v_0

        int j = 0;
        if (Model_type == 1) {
            para(5) = std::exp(para_tilde(5));  // shape
            j = 1;
        }
        if (Model_type == 2) {
            para(5) = para_tilde(5);   // xi
            para(6) = para_tilde(6);   // varphi
            para(7) = para_tilde(7);   // delta_1
            para(8) = para_tilde(8);   // delta_2
            para(9) = std::exp(para_tilde(9));  // shape
            j = 5;
        }
        if (LEVIER) {
            para(5 + j) = std::exp(para_tilde(5 + j));        // l
            para(6 + j) = 1.0 / (1.0 + std::exp(para_tilde(6 + j)));  // theta
        }

        return para;
    }

    // Natural to working parameter transformation
    VectorXd natWork(const VectorXd& para, bool LEVIER = false,
                     int Model_type = 0) {
        VectorXd para_tilde = para;

        para_tilde[0] = std::log((1.0 / para[0]) - 1.0);  // omega
        para_tilde[1] = std::log((1.0 / para[1]) - 1.0);  // a
        para_tilde[2] = std::log(para[2] - 1.0);          // b
        para_tilde[3] = std::log(para[3]);                // sigma
        para_tilde[4] = std::log((1.0 / para[4]) - 1.0);  // v_0

        int j = 0;
        if (Model_type == 1) {
            para_tilde[5] = std::log(para[5]);  // shape
            j = 1;
        }
        if (Model_type == 2) {
            para_tilde[5] = para[5];    // xi
            para_tilde[6] = para[6];    // varphi
            para_tilde[7] = para[7];    // delta_1
            para_tilde[8] = para[8];    // delta_2
            para_tilde[9] = std::log(para[9]);  // shape
            j = 5;
        }
        if (LEVIER) {
            para_tilde[5 + j] = std::log(para[5 + j]);  // l
            para_tilde[6 + j] = std::log((1.0 / para[6 + j]) - 1.0);  // theta
        }

        return para_tilde;
    }

    // Compute volatility vector
    VectorXd volatilityVector(const VectorXd& para, int K, int N) {
        double omega = para[0];
        double v0 = para[4];
        double sigma2 = para[3];

        // Create state vector
        VectorXd sigma_i(K);
        for (int k = 0; k < K; k++) {
            sigma_i[k] = v0 * std::pow((2.0 - v0) / v0, k);
        }

        // Compute probabilities
        VectorXd probaPi = computeBinomialProbs(omega, K);
        double e_i = probaPi.dot(sigma_i);
        sigma_i = sigma_i / e_i;

        // Kronecker product for N components
        VectorXd sigma = VectorXd::Ones(1);
        for (int i = 0; i < N; i++) {
            sigma = kroneckerProduct(sigma, sigma_i);
        }

        return sigma2 * sigma;
    }

    // Compute transition matrix
    MatrixXd transitionMatrix(const VectorXd& para, int K, int N) {
        double omega = para[0];
        double a = para[1];
        double b = para[2];

        // Compute persistence levels
        VectorXd phi(N);
        for (int i = 0; i < N; i++) {
            phi[i] = std::pow(a, std::pow(b, i));
        }

        // Build transition matrices for each component
        std::vector<MatrixXd> P_components;
        VectorXd probaPi = computeBinomialProbs(omega, K);

        for (int i = 0; i < N; i++) {
            MatrixXd P_i = phi[i] * MatrixXd::Identity(K, K) +
                          (1 - phi[i]) * VectorXd::Ones(K) * probaPi.transpose();
            P_components.push_back(P_i);
        }

        // Kronecker product of all components
        MatrixXd P = P_components[0];
        for (int i = 1; i < N; i++) {
            P = kroneckerProduct(P, P_components[i]);
        }

        return P;
    }

    // Compute log-likelihood
    double logLikelihood(const VectorXd& para_tilde, const MatrixXd& data,
                        int Model_type = 0, bool LEVIER = false) {
        VectorXd para = workNat(para_tilde, LEVIER, Model_type);

        int n = data.rows();
        VectorXd sigma = volatilityVector(para, K_, N_);
        MatrixXd P = transitionMatrix(para, K_, N_);
        VectorXd p0 = computeStationaryDist(para[0], K_, N_);

        double loglik = 0.0;
        VectorXd filter_probs = p0;

        // Forward filtering
        for (int t = 0; t < n; t++) {
            VectorXd likelihood_t = computeLikelihood(data.row(t), sigma, para,
                                                      Model_type, LEVIER);
            VectorXd joint_probs = filter_probs.array() * likelihood_t.array();
            double normalizer = joint_probs.sum();
            loglik += std::log(normalizer);

            // Update filter probabilities
            filter_probs = joint_probs / normalizer;
            if (t < n - 1) {
                filter_probs = P.transpose() * filter_probs;
            }
        }

        return -loglik;  // Return negative for minimization
    }

private:
    int K_;
    int N_;
    int state_size_;

    // Helper function: Compute binomial probabilities
    VectorXd computeBinomialProbs(double omega, int K) {
        VectorXd probs(K);
        for (int k = 0; k < K; k++) {
            probs[k] = binomialCoeff(K - 1, k) *
                      std::pow(omega, k) * std::pow(1 - omega, K - 1 - k);
        }
        return probs;
    }

    // Helper function: Binomial coefficient
    double binomialCoeff(int n, int k) {
        if (k > n) return 0;
        if (k == 0 || k == n) return 1;

        double result = 1;
        for (int i = 0; i < k; i++) {
            result *= (n - i);
            result /= (i + 1);
        }
        return result;
    }

    // Helper function: Kronecker product
    MatrixXd kroneckerProduct(const MatrixXd& A, const MatrixXd& B) {
        int m = A.rows(), n = A.cols();
        int p = B.rows(), q = B.cols();

        MatrixXd result(m * p, n * q);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                result.block(i * p, j * q, p, q) = A(i, j) * B;
            }
        }
        return result;
    }

    VectorXd kroneckerProduct(const VectorXd& a, const VectorXd& b) {
        int m = a.size();
        int n = b.size();
        VectorXd result(m * n);

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                result[i * n + j] = a[i] * b[j];
            }
        }
        return result;
    }

    // Compute stationary distribution
    VectorXd computeStationaryDist(double omega, int K, int N) {
        VectorXd proba_single = computeBinomialProbs(omega, K);
        VectorXd proba = proba_single;

        for (int i = 1; i < N; i++) {
            proba = kroneckerProduct(proba, proba_single);
        }

        return proba;
    }

    // Compute likelihood for one observation
    VectorXd computeLikelihood(const VectorXd& obs, const VectorXd& sigma,
                               const VectorXd& para, int Model_type, bool LEVIER) {
        int n_states = sigma.size();
        VectorXd likelihood(n_states);

        if (Model_type == 0) {
            // Univariate returns model
            double r = obs[0];
            for (int i = 0; i < n_states; i++) {
                double std_dev = std::sqrt(sigma[i]);
                likelihood[i] = normalPDF(r, 0, std_dev);
            }
        } else if (Model_type == 1) {
            // Univariate realized variance model
            double rv = obs[0];
            double shape = para[5];
            for (int i = 0; i < n_states; i++) {
                likelihood[i] = gammaPDF(rv / sigma[i], shape, 1.0 / shape) / sigma[i];
            }
        } else if (Model_type == 2) {
            // Joint model
            double r = obs[0];
            double rv = obs[1];
            double xi = para[5];
            double varphi = para[6];
            double delta1 = para[7];
            double delta2 = para[8];
            double shape = para[9];

            for (int i = 0; i < n_states; i++) {
                double std_dev = std::sqrt(sigma[i]);
                double epsilon = r / std_dev;
                double mu_rv = xi + varphi * std::log(sigma[i]) +
                              delta1 * epsilon + delta2 * (epsilon * epsilon - 1);

                likelihood[i] = normalPDF(r, 0, std_dev) *
                               lognormalPDF(rv, mu_rv, std::sqrt(shape));
            }
        }

        return likelihood;
    }

    // PDF functions
    double normalPDF(double x, double mean, double std_dev) {
        double z = (x - mean) / std_dev;
        return std::exp(-0.5 * z * z) / (std_dev * std::sqrt(2 * M_PI));
    }

    double gammaPDF(double x, double shape, double scale) {
        if (x <= 0) return 0;
        return std::pow(x, shape - 1) * std::exp(-x / scale) /
               (std::pow(scale, shape) * std::tgamma(shape));
    }

    double lognormalPDF(double x, double mu, double sigma) {
        if (x <= 0) return 0;
        double log_x = std::log(x);
        double z = (log_x - mu) / sigma;
        return std::exp(-0.5 * z * z) / (x * sigma * std::sqrt(2 * M_PI));
    }
};