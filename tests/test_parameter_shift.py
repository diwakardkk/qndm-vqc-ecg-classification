import numpy as np

from quantum.ansatz import CircuitSpec
from quantum.gradients import analytic_sample_gradient, parameter_shift_sample_gradient
from quantum.observables import observable_for_regime


def test_parameter_shift_matches_analytic():
    spec = CircuitSpec(3, 2)
    obs = observable_for_regime("A", 3)
    rng = np.random.default_rng(7)
    x = rng.normal(size=3)
    theta = rng.normal(scale=0.2, size=spec.n_parameters)
    exact = analytic_sample_gradient(x, theta, spec, obs)
    ps = parameter_shift_sample_gradient(x, theta, spec, obs)
    np.testing.assert_allclose(ps, exact, atol=1e-10)

