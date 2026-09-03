import numpy as np

from quantum.ansatz import CircuitSpec
from quantum.gradients import analytic_sample_gradient
from quantum.observables import observable_for_regime
from quantum.qndm import qndm_sample_gradient, qndm_uses_detector_qubit


def test_qndm_detector_exists():
    spec = CircuitSpec(2, 1)
    obs = observable_for_regime("A", 2)
    x = np.array([0.1, -0.2])
    theta = np.array([0.03, -0.04])
    assert qndm_uses_detector_qubit(x, theta, spec, obs)


def test_qndm_small_lambda_matches_reference():
    spec = CircuitSpec(2, 1)
    obs = observable_for_regime("A", 2)
    x = np.array([0.2, -0.3])
    theta = np.array([0.08, -0.12])
    exact = analytic_sample_gradient(x, theta, spec, obs)
    est = qndm_sample_gradient(x, theta, spec, obs, lam=1e-4)
    np.testing.assert_allclose(est, exact, atol=1e-6)

