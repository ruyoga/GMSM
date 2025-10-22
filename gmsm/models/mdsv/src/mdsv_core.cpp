// mdsv_core.cpp - Core C++ implementation for MDSV with leverage effect
// FIXED VERSION - Corrects leverage application in likelihood computation
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>

using namespace Eigen;

class MDSVCore {
private:
    int K_;  // States per component
    int N_;  // Number of components
    int state_size_;  // Total number of states (K^N)

public:
    // Constructor
    MDSVCore(int K, int N) : K_(K), N_(N) {
        state_size_ = static_cast<int>(std::pow(K, N));
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
            para(5 + j) = std::exp(para_tilde(5 + j));        // l1
            para(6 + j) = 1.0 / (1.0 + std::exp(para_tilde(6 + j)));  // theta_l
        }

        return para;
    }

    // Natural to working parameter transformation
    VectorXd natWork(const VectorXd& para, bool LEVIER = false,
                     int Model_type = 0) {
        VectorXd para_tilde = para;

        para_tilde(0) = std::log((1.0 / para(0)) - 1.0);  // omega
        para_tilde(1) = std::log((1.0 / para(1)) - 1.0);  // a
        para_tilde(2) = std::log(para(2) - 1.0);          // b
        para_tilde(3) = std::log(para(3));                // sigma
        para_tilde(4) = std::log((1.0 / para(4)) - 1.0);  // v_0

        int j = 0;
        if (Model_type == 1) {
            para_tilde(5) = std::log(para(5));  // shape
            j = 1;
        }
        if (Model_type == 2) {
            para_tilde(5) = para(5);    // xi
            para_tilde(6) = para(6);    // varphi
            para_tilde(7) = para(7);    // delta_1
            para_tilde(8) = para(8);    // delta_2
            para_tilde(9) = std::log(para(9));  // shape
            j = 5;
        }
        if (LEVIER) {
            para_tilde(5 + j) = std::log(para(5 + j));  // l1
            para_tilde(6 + j) = std::log((1.0 / para(6 + j)) - 1.0);  // theta_l
        }

        return para_tilde;
    }

    // Compute volatility vector
    VectorXd volatilityVector(const VectorXd& para, int K, int N) {
        double omega = para(0);
        double v0 = para(4);
        double sigma2 = para(3);

        // Create state vector
        VectorXd sigma_i(K);
        for (int k = 0; k < K; k++) {
            sigma_i(k) = v0 * std::pow((2.0 - v0) / v0, k);
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
        double omega = para(0);
        double a = para(1);
        double b = para(2);

        // Compute persistence levels
        VectorXd phi(N);
        for (int i = 0; i < N; i++) {
            phi(i) = std::pow(a, std::pow(b, i));
        }

        // Stationary probabilities
        VectorXd proba_pi = computeBinomialProbs(omega, K);

        // Build transition matrices for each component
        std::vector<MatrixXd> P_components;
        for (int i = 0; i < N; i++) {
            MatrixXd P_i = phi(i) * MatrixXd::Identity(K, K) +
                          (1.0 - phi(i)) * VectorXd::Ones(K) * proba_pi.transpose();
            P_components.push_back(P_i);
        }

        // Kronecker product of all components
        MatrixXd P = P_components[0];
        for (int i = 1; i < N; i++) {
            P = kroneckerProduct(P, P_components[i]);
        }

        return P;
    }

    // Compute leverage multipliers Lt from returns
    VectorXd computeLeverageMultipliers(const VectorXd& returns,
                                         double l1,
                                         double theta_l,
                                         int NL = 70) {
        int T = returns.size();
        VectorXd L = VectorXd::Ones(T);

        // Recursive computation of Lt
        for (int t = 1; t < T; t++) {
            double Lt = 1.0;
            int max_lag = std::min(t, NL);

            for (int i = 1; i <= max_lag; i++) {
                if (returns(t - i) < 0) {
                    double li = l1 * std::pow(theta_l, i - 1);
                    Lt *= (1.0 + li * std::abs(returns(t - i)) / std::sqrt(L(t - i)));
                }
            }
            L(t) = Lt;
        }

        return L;
    }

    // FIXED: Compute observation likelihood with leverage properly applied
    VectorXd computeLikelihood(const VectorXd& obs,
                                const VectorXd& sigma_base,
                                const VectorXd& para,
                                int Model_type,
                                bool LEVIER,
                                double L_t = 1.0) {
        int n_states = sigma_base.size();
        VectorXd likelihood = VectorXd::Zero(n_states);

        if (Model_type == 0) {
            // Univariate returns model
            double r = obs(0);
            for (int i = 0; i < n_states; i++) {
                // FIXED: Apply leverage to volatility
                double sigma_leveraged = sigma_base(i) * L_t;
                double std_dev = std::sqrt(sigma_leveraged);
                likelihood(i) = normalPDF(r, 0, std_dev);
            }
        } else if (Model_type == 1) {
            // Univariate realized variance model
            double rv = obs(0);
            double shape = para(5);
            for (int i = 0; i < n_states; i++) {
                // FIXED: Apply leverage to volatility for RV model
                double sigma_leveraged = sigma_base(i) * L_t;
                likelihood(i) = gammaPDF(rv / sigma_leveraged, shape, 1.0 / shape) / sigma_leveraged;
            }
        } else if (Model_type == 2) {
            // Joint model
            double r = obs(0);
            double rv = obs(1);
            double xi = para(5);
            double varphi = para(6);
            double delta1 = para(7);
            double delta2 = para(8);
            double shape = para(9);

            for (int i = 0; i < n_states; i++) {
                // CRITICAL FIX: Apply leverage to base volatility
                double sigma_leveraged = sigma_base(i) * L_t;
                double std_dev = std::sqrt(sigma_leveraged);
                double epsilon = r / std_dev;

                // CRITICAL FIX: Use leveraged volatility in log(RV) equation
                // This is Equation 21 from the paper with V^(L)_t
                double mu_rv = xi + varphi * std::log(sigma_leveraged) +
                              delta1 * epsilon + delta2 * (epsilon * epsilon - 1.0);

                likelihood(i) = normalPDF(r, 0, std_dev) *
                               lognormalPDF(rv, mu_rv, std::sqrt(shape));
            }
        }

        return likelihood;
    }

    // CRITICAL FIX: Log-likelihood with proper leverage application
    double logLikelihood(const VectorXd& para_tilde,
                         const MatrixXd& data,
                         int Model_type = 0,
                         bool LEVIER = false) {
        // Transform parameters to natural scale
        VectorXd para = workNat(para_tilde, LEVIER, Model_type);

        int T = data.rows();

        // Get model components - base volatility without leverage
        VectorXd sigma_base = volatilityVector(para, K_, N_);
        MatrixXd P = transitionMatrix(para, K_, N_);
        VectorXd pi0 = computeStationaryDist(para(0), K_, N_);

        // CRITICAL FIX: Compute leverage multipliers from returns
        VectorXd L = VectorXd::Ones(T);
        if (LEVIER && (Model_type == 0 || Model_type == 2)) {
            int j = (Model_type == 2) ? 5 : 0;
            double l1 = para(5 + j);
            double theta_l = para(6 + j);

            // Extract returns column
            VectorXd returns = data.col(0);
            L = computeLeverageMultipliers(returns, l1, theta_l);
        }

        // Forward filtering with leverage properly applied
        double log_lik = 0.0;
        VectorXd alpha = pi0;

        for (int t = 0; t < T; t++) {
            // CRITICAL FIX: Pass base volatility and leverage multiplier separately
            VectorXd likelihood = computeLikelihood(
                data.row(t).transpose(),
                sigma_base,  // Base volatility
                para,
                Model_type,
                LEVIER,
                L(t)  // Leverage multiplier for time t
            );

            // Update step
            alpha = alpha.array() * likelihood.array();
            double c = alpha.sum();

            if (c > 1e-300) {
                alpha = alpha / c;
                log_lik += std::log(c);
            } else {
                // Numerical issues - return large negative value
                return -1e10;
            }

            // Predict step for next iteration
            if (t < T - 1) {
                alpha = P.transpose() * alpha;
            }
        }

        // Return negative log-likelihood for minimization
        return -log_lik;
    }

private:
    // Helper function: Binomial probabilities
    VectorXd computeBinomialProbs(double omega, int K) {
        VectorXd probs(K);
        for (int k = 0; k < K; k++) {
            probs(k) = binomialCoeff(K - 1, k) *
                      std::pow(omega, k) * std::pow(1.0 - omega, K - 1 - k);
        }
        return probs;
    }

    // Helper function: Binomial coefficient
    double binomialCoeff(int n, int k) {
        if (k > n) return 0.0;
        if (k == 0 || k == n) return 1.0;

        double result = 1.0;
        for (int i = 0; i < k; i++) {
            result *= (n - i);
            result /= (i + 1);
        }
        return result;
    }

    // Helper function: Kronecker product for matrices
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

    // Helper function: Kronecker product for vectors
    VectorXd kroneckerProduct(const VectorXd& a, const VectorXd& b) {
        int m = a.size();
        int n = b.size();
        VectorXd result(m * n);

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                result(i * n + j) = a(i) * b(j);
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

    // PDF functions
    double normalPDF(double x, double mean, double std_dev) {
        if (std_dev <= 0) return 0.0;
        double z = (x - mean) / std_dev;
        return std::exp(-0.5 * z * z) / (std_dev * std::sqrt(2.0 * M_PI));
    }

    double gammaPDF(double x, double shape, double scale) {
        if (x <= 0 || shape <= 0 || scale <= 0) return 0.0;
        return std::pow(x, shape - 1.0) * std::exp(-x / scale) /
               (std::pow(scale, shape) * std::tgamma(shape));
    }

    double lognormalPDF(double x, double mu, double sigma) {
        if (x <= 0 || sigma <= 0) return 0.0;
        double log_x = std::log(x);
        double z = (log_x - mu) / sigma;
        return std::exp(-0.5 * z * z) / (x * sigma * std::sqrt(2.0 * M_PI));
    }
};