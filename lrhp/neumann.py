"""
Truncated Neumann approximation.
"""
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, grad
from jax.experimental import optimizers
from tqdm import tqdm

from . import util


def neumann_approx(W, H, k):
    d = W.shape[1]
    approx = jnp.eye(d) + W.T.dot(H)

    Z0 = H.dot(W.T)
    for j in range(2, k + 1):
        Z = Z0
        for _ in range(j - 2):
            Z = Z.dot(Z0)
        approx = approx + W.T.dot(Z).dot(H)

    return approx


def symm_neumann_approx(W, k):
    return neumann_approx(W, W, k)


class TruncatedNeumannEstimator:
    """
    Approximate letting Psi = I + Phi + Phi^2 + ... + Phi^k
    """

    def __init__(
        self,
        rank: int,
        approx_degree: int = 2,
        is_nmf: bool = False,
        integral_support: float = 10.0,
        is_symmetric: bool = True,  # not used
    ) -> None:
        self.rank = rank
        self.is_nmf = is_nmf
        self.is_symmetric = is_symmetric
        self.approx_degree = approx_degree
        self.integral_support = integral_support

    @staticmethod
    def _get_C_lam(t, c, h, T=None):
        if T is None:
            T = t[-1]

        C = util.get_empirical_covariance(t, c, h=h, T=T)
        lam = np.bincount(c) / T

        return C, lam

    def fit_cumulants(
        self, C, lam, learning_rate=1e-5, num_epochs=1000,
    ) -> Tuple[np.ndarray, float]:
        r, d = self.rank, lam.shape[0]

        @jit
        def loss(W_in):
            W_out = jax.nn.softplus(W_in) if self.is_nmf else W_in
            Psi = neumann_approx(W_out, W_out, k=self.approx_degree)
            loss_val = jnp.linalg.norm(
                C - (Psi * lam).dot(Psi.T), 'fro'
            )
            return loss_val

        opt_init, opt_update, get_params = optimizers.adam(learning_rate)
        @jit
        def update(i, state):
            params = get_params(state)
            return opt_update(i, grad(loss)(params), state)

        W0 = np.random.rand(r, d)
        W0 /= np.linalg.norm(W0)
        opt_state = opt_init(W0)
        W = (
            W0 if not self.is_nmf
            else np.log(np.expm1(W0))
        )

        tq_iterator = tqdm(range(num_epochs))

        for epoch in tq_iterator:
            opt_state = update(epoch, opt_state)

            W = get_params(opt_state)
            tq_iterator.set_postfix({"Loss": loss(W)})

        return jax.nn.softplus(W) if self.is_nmf else W, loss

    def fit(self, t, c, *, T=None, **kwargs) -> Tuple[np.ndarray, float]:
        C, lam = self._get_C_lam(t, c, T=T, h=self.integral_support)
        return self.fit_cumulants(C, lam, **kwargs)
