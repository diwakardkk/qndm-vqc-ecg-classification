from quantum.ansatz import CircuitSpec
from quantum.observables import observable_for_regime
from quantum.resource_counter import analytic_shots, circuit_resources


def test_resource_equations():
    spec = CircuitSpec(4, 2)
    obs = observable_for_regime("A", 4)
    resources = circuit_resources(spec, obs)
    assert resources.parameters == 8
    assert resources.k == 4 + (2 * 4 - 1) * 2
    shots = analytic_shots(10, spec, obs, epochs=3, m_f=100, m_d=100, m_q=100)
    assert shots["dm_gradient_per_sample"] == 2 * obs.g_m * spec.n_parameters * 100
    assert shots["qndm_gradient_per_sample"] == spec.n_parameters * 100

