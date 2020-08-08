import numpy as np

from lrhp.neumann import symm_neumann_approx, neumann_approx


def test_symm_approx_correct():
    W = np.random.rand(2, 10)

    I = np.eye(W.shape[1])
    Phi = W.T.dot(W)

    Approx_2 = np.array(symm_neumann_approx(W, 2))
    Approx_3 = np.array(symm_neumann_approx(W, 3))
    Approx_4 = np.array(symm_neumann_approx(W, 4))

    assert np.allclose(Approx_2, I + Phi + Phi @ Phi)
    assert np.allclose(Approx_3, I + Phi + Phi @ Phi + Phi @ Phi @ Phi)
    assert np.allclose(
        Approx_4, I + Phi + Phi @ Phi + Phi @ Phi @ Phi + Phi @ Phi @ Phi @ Phi
    )


def test_nosymm_approx_correct():
    W = np.random.rand(2, 10)
    H = np.random.rand(2, 10)

    I = np.eye(W.shape[1])
    Phi = W.T.dot(H)

    Approx_2 = np.array(neumann_approx(W, H, 2))
    Approx_3 = np.array(neumann_approx(W, H, 3))
    Approx_4 = np.array(neumann_approx(W, H, 4))

    assert np.allclose(Approx_2, I + Phi + Phi @ Phi)
    assert np.allclose(Approx_3, I + Phi + Phi @ Phi + Phi @ Phi @ Phi)
    assert np.allclose(
        Approx_4, I + Phi + Phi @ Phi + Phi @ Phi @ Phi + Phi @ Phi @ Phi @ Phi
    )
