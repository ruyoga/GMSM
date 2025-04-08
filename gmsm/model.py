import numpy as np

def msm_likelihood(pimat0, omegat, A):
    """
    log likelihood and filtered probs
    :param pimat0: initial state probs (1 x k) matrix
    :param omegat: conditional densities (T-1 x k) matrix
    :param A: transition matrix (k x k) matrix
    :return: dict of filtered probabilities (pmat)
             negative log likelihood (ll)
             vector of log-likelihoods (lls)
    """
    T = omegat.shape[0] + 1
    k = omegat.shape[1]

    pmat = np.zeros((T, k))
    LLs = np.zeros(T - 1)

    pmat[0, :] = pimat0

    for i in range(1, T):
        piA = pmat[i-1, :] @ A
        pinum = omegat[i-1, :] * piA
        pidenom = np.sum(omegat[i-1, :] * piA)

        if pidenom == 0:
            pmat[i, 0] = 1
        else:
            pmat[i, :] = pinum / pidenom

        LLs[i-1] = np.log(pidenom)

    ll = -np.sum(LLs)

    return {
        "pmat": pmat,
        "LL": ll,
        "LLs": LLs
    }

def msm_ll(pimat0, omegat, A):
    """
    calculate only the log likelihood
    :param pimat0: initial state probs (1 x k) matrix
    :param omegat: conditional densities (T-1 x k) matrix
    :param A: transition matrix (k x k) matrix
    :return: negative log likelihood
    """
    T = omegat.shape[0] + 1
    k = omegat.shape[1]

    pmat = np.zeros((T, k))
    LLs = np.zeros(T - 1)

    pmat[0, :] = pimat0

    for i in range(1, T):
        piA = pmat[i-1, :] @ A
        pinum = omegat[i-1, :] * piA
        pidenom = np.sum(omegat[i-1, :] * piA)

        if pidenom == 0:
            pmat[i, 0] = 1
        else:
            pmat[i, :] = pinum / pidenom

        LLs[i-1] = np.log(pidenom)

    ll = -np.sum(LLs)

    return ll

def msm_smooth(A, P):
    """
    function to smoothen transition matrix
    :param A: transition matrix (k x k) matrix
    :param P: filtered probs (T x k) matrix
    :return:
    """
    T = P.shape[0]
    k = P.shape[1]

    p = np.zeros((T, k))
    pt = np.zeros((T, k))

    # Forward pass
    pt[0, :] = np.ones(k) / k
    for t in range(1, T):
        pt[t, :] = A.T @ P[t-1, :]

    # Backward pass
    p[T-1, :] = P[T-1, :]

    for t in range(T-2, -1, -1):
        for i in range(k):
            total = 0
            for j in range(k):
                if pt[t+1, j] > 0:
                    total += A[i, j] * p[t+1, j] / pt[t+1, j]
            p[t, i] = P[t, i] * total

    return p