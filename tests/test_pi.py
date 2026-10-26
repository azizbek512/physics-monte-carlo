"""Task 1 spetsifikatsiyasi. Bu testlarni o'zgartirma — kodingni ularga moslashtir."""
import numpy as np
import pytest

from mc.pi import estimate_pi


def test_returns_estimate_and_error():
    est, err = estimate_pi(10_000, np.random.default_rng(0))
    assert isinstance(est, float) and isinstance(err, float)
    assert err > 0


def test_close_to_pi_within_error():
    est, err = estimate_pi(1_000_000, np.random.default_rng(1))
    assert abs(est - np.pi) < 5 * err


def test_reproducible_with_same_seed():
    a = estimate_pi(10_000, np.random.default_rng(42))
    b = estimate_pi(10_000, np.random.default_rng(42))
    assert a == b


def test_error_scales_as_inverse_sqrt_n():
    _, err_small = estimate_pi(10_000, np.random.default_rng(2))
    _, err_large = estimate_pi(1_000_000, np.random.default_rng(3))
    # N 100 marta oshsa, xato ~10 marta kamayishi kerak
    assert err_small / err_large == pytest.approx(10, rel=0.1)
