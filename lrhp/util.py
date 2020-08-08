from typing import List, Tuple

import numpy as np
import scipy.linalg as spla

from numba import jit as numbajit


@numbajit
def mv_diff_count(lag: np.ndarray, c: np.ndarray, h: float) -> np.ndarray:
    N, K = len(lag), np.max(c) + 1

    t = np.cumsum(lag)
    T = t[-1] + 1e-3
    lam = np.bincount(c) / T
    u = 0  # index to keep track of beginning of window

    out = np.zeros((K, K))

    for i in range(N):
        if t[i] < h or t[i] > T - h:
            continue

        while t[u] < t[i] - h:
            u += 1
        j = u

        while t[j] < t[i] + h:
            out[c[i], c[j]] += 1
            j += 1

        for k in range(K):
            out[c[i], k] -= 2 * h * lam[k]

    return out


def get_empirical_covariance(t, c, h, T=None):
    """
    Given a sequence of events, compute the empirical covariance
    matrix.
    
    Parameters
    ----------
    t: numpy.ndarray
        An array of timestamps of events shape (n_events,). The 
        timestamps must be in increasing order.
    c: numpy.ndarray
        Array of marks, with data type `int`. Shape (n_events,).
    h: float
        The horizon of the integrated cumulant. See [Achab et al., 2019]
        Usually on the same order of the average delay.
    T: float
        End of the observation window.
        
    Returns
    -------
    C: numpy.ndarray
        Empirical covariance matrix, shape (n_marks, n_marks)
    """
    if T is None:
        T = t[-1]

    C = mv_diff_count(
        np.diff(np.r_[0, t]), c, h
    ) / T
    return 0.5 * (C + C.T)


def get_full_moment_based_phi(C, lam):
    I = np.eye(C.shape[0])
    lamr = np.sqrt(lam)

    DCD = np.transpose((C * lamr).T * lamr)
    s, V = spla.eigh(DCD)

    s2 = np.clip(1 / s, a_min=0, a_max=None)

    R = (V * np.sqrt(s2)).dot(V.T)

    DRD = np.transpose((R * lamr).T * lamr)

    return np.clip(I - DRD, a_min=0, a_max=None)


def get_low_rank_eigh(Phi, r=5):
    s, V = spla.eigh(Phi)
    Ve = V[:, -r:]  # major eigenvectors

    return (Ve * s[-r:]**.5).T


def dMd(d1, M, d2):
    return (M*d2)*d1[:, None]


def MdM(M1, d, M2):
    return (M1*d) @ M2


def L(A):
    """Graph Laplacian"""
    return np.diag(A.sum(1)) - A


def get_numerical_rank(A):
    s = np.linalg.eigvals(A)
    return np.sum(np.abs(s) > 1e-4)


def get_lr_approx(A, r=10, direction="largest"):
    s, V = np.linalg.eigh(A)
    sl = (
        slice(-r, None) if direction == "largest" else
        slice(None, r)
    )
    return MdM(
        V[:, sl], s[sl], V[:, sl].T
    )


def tick_data_to_hawkeslib(data: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a `tick` format dataset to `hawkeslib` format

    Parameters
    ----------
    data
        Data in `tick` format, a list of timestamp arrays, all corresponding to
        a mark

    Returns
    -------
    t: np.ndarray
        Timestamps in sorted order
    c: np.ndarray
        Marks in sorted order, where each mark refers to the index of the original
        list (`tick` data).
    """
    t_c_zip = zip(*[(t, np.ones_like(t, dtype=int) * c) for c, t in enumerate(data)])
    all_ts, all_cs = (np.concatenate(x) for x in t_c_zip)
    sortix = np.argsort(all_ts)
    return all_ts[sortix], all_cs[sortix]


def hawkeslib_data_to_tick(t, c):
    data = []
    for i in range(c.max() + 1):
        data.append(t[c == i])
    return data
