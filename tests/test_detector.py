import numpy as np

from quantum.qndm import coherence_from_xy


def test_detector_xy_coherence_mapping():
    rho = 0.25 - 0.1j
    x_exp = 2 * rho.real
    y_exp = -2 * rho.imag
    assert coherence_from_xy(x_exp, y_exp) == rho

