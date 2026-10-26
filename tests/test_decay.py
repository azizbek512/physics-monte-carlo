"""Task 2 spetsifikatsiyasi. Bu testlarni o'zgartirma — kodingni ularga moslashtir."""
import numpy as np
import pytest

from mc.decay import simulate_decay_steps, sample_decay_times, mle_lambda


def test_steps_shape_and_start():
    n = simulate_decay_steps(n0=1000, lam=0.1, dt=0.1, n_steps=50, rng=np.random.default_rng(0))
    assert len(n) == 51
    assert n[0] == 1000


def test_steps_monotonic_non_increasing():
    n = simulate_decay_steps(n0=5000, lam=0.5, dt=0.05, n_steps=200, rng=np.random.default_rng(1))
    assert np.all(np.diff(n) <= 0)
    assert np.all(n >= 0)


def test_steps_follow_exponential_law():
    n0, lam, dt, steps = 100_000, 0.2, 0.01, 500
    n = simulate_decay_steps(n0, lam, dt, steps, np.random.default_rng(2))
    t_end = dt * steps
    assert n[-1] / n0 == pytest.approx(np.exp(-lam * t_end), rel=0.05)


def test_sampled_times_mean_and_positive():
    lam = 2.0
    t = sample_decay_times(200_000, lam, np.random.default_rng(3))
    assert np.all(t >= 0)
    assert t.mean() == pytest.approx(1 / lam, rel=0.01)


def test_mle_lambda_recovers_true_value():
    lam = 0.7
    t = sample_decay_times(100_000, lam, np.random.default_rng(4))
    lam_hat, lam_err = mle_lambda(t)
    assert lam_err > 0
    assert abs(lam_hat - lam) < 5 * lam_err
