import numpy as np
import itertools

def msm_A(b, gamma_kbar, kbar):
    """
    Calculate transition matrix
    :param b: growth rate of switching probs
    :param gamma_kbar: transition prob of highest frequency component
    :param kbar: n frequency components
    :return: transition matrix (2^kbar x 2^kbar)
    """
    gamma_k = np.zeros(kbar)
    gamma_k[0] = 1 - (1 - gamma_kbar) ** (1 / (b ** (kbar - 1)))

    # Initial 2x2 transition matrix
    A = np.array([
        [1 - gamma_k[0] + 0.5 * gamma_k[0], 0.5 * gamma_k[0]],
        [0.5 * gamma_k[0], 1 - gamma_k[0] + 0.5 * gamma_k[0]]
    ])

    if kbar > 1:
        for i in range(1, kbar):
            gamma_k[i] = 1 - (1 - gamma_k[0]) ** (b ** (i - 1))
            a = np.array([
                [1 - gamma_k[i] + 0.5 * gamma_k[i], 0.5 * gamma_k[i]],
                [0.5 * gamma_k[i], 1 - gamma_k[i] + 0.5 * gamma_k[i]]
            ])
            # Kronecker product
            A = np.kron(A, a)

    return A

def msm_clustermat(m0, kbar):
    """
    Calculate matrix of volatility components
    :param m0: value of each volaility component
    :param kbar: number of frequency components
    :return: grid of volatility components (2^kbar x 2^kbar)
    """

    # Create all possible combinations of m0 and 2-m0
    m_values = [m0, 2-m0]
    combinations = list(itertools.product(*[m_values] * kbar))

    # Convert to matrix and reverse the order of columns
    M_mat = np.array(combinations)
    M_mat = M_mat[:, ::-1]

    return M_mat

def msm_states(m0, kbar):
    """
    Calculate all possible state values
    :param m0: state variable value
    :param kbar: number of frequency components
    :return: state vector (1 x 2^kbar)
    """
    m1 = 2 - m0
    k_2 = 2 ** kbar
    g_m = np.zeros(k_2)

    for i in range(k_2):
        g = 1
        for j in range(kbar):
            if (i & (1 << j)) != 0:
                g *= m1
            else:
                g *= m0
        g_m[i] = g

    g_m = np.sqrt(g_m).reshape(1, -1)
    return g_m

def msm_mat_power(A, power):
    """
    calculate matrix power
    :param A: square matrix
    :param power: power to raise matrix to
    :return: A^power
    """
    return np.linalg.matrix_power(A, power)

def msm_marginals(p, m, Mmat, kbar):
    """
    Calculqte marginal probabilities for msm components
    :param p: matrix of smoothed/filtered state probs
    :param m: value of volaility component
    :param Mmat: matrix of volatility components from clustermat()
    :param kbar: number of frequency components
    :return: matrix of marginal probabilities (T x kbar)
    """
    Mmat = Mmat.T
    m_marginals = np.zeros((p.shape[0], kbar))

    for k in range(kbar):
        col_sel = (Mmat[k, :] == m)
        # Sum probabilities for these states
        m_marginals[:, k] = np.sum(p[:, col_sel], axis=1)

    return m_marginals