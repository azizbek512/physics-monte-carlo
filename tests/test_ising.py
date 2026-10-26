"""Task 3 spetsifikatsiyasi (2D Ising Model). Bu testlarni o'zgartirma."""
import numpy as np
import pytest

from mc.ising import lattice_energy, delta_energy, metropolis_sweep


def test_all_aligned_energy():
    # L x L panjarada hamma spinlar +1 bo'lsa, har bir katakda 2 ta bog'lanish (o'ng va past) -> E = -2 * J * L^2
    L = 6
    spins = np.ones((L, L), dtype=int)
    assert lattice_energy(spins, J=1.0) == pytest.approx(-2.0 * L * L)
    assert lattice_energy(-spins, J=1.0) == pytest.approx(-2.0 * L * L)


def test_checkerboard_energy():
    # Shaxmat doskasi (+1, -1 almashib kelsa) -> barcha qo'shnilar qarama-qarshi -> E = +2 * J * L^2
    L = 6
    idx = np.indices((L, L)).sum(axis=0)
    spins = np.where(idx % 2 == 0, 1, -1)
    assert lattice_energy(spins, J=1.0) == pytest.approx(2.0 * L * L)


def test_delta_energy_matches_full_energy_difference():
    rng = np.random.default_rng(0)
    L = 8
    spins = rng.choice([-1, 1], size=(L, L))
    e_before = lattice_energy(spins, J=1.5)

    for i in [0, 3, L - 1]:
        for j in [0, 4, L - 1]:
            dE = delta_energy(spins, i, j, J=1.5)
            flipped = spins.copy()
            flipped[i, j] *= -1
            e_after = lattice_energy(flipped, J=1.5)
            assert dE == pytest.approx(e_after - e_before)


def test_low_temp_orders_ferromagnetically():
    # T = 1.0 << Tc (~2.269): barcha spinlar bir xil yo'nalishga kelishi kerak (|m| -> 1)
    rng = np.random.default_rng(42)
    L = 10
    spins = np.ones((L, L), dtype=int)
    # Bir nechta spinni buzib qo'yamiz
    spins[0, 0] = -1
    spins[5, 5] = -1

    for _ in range(50):
        spins = metropolis_sweep(spins, T=1.0, rng=rng)

    mag = abs(spins.mean())
    assert mag > 0.9


def test_high_temp_disorders_paramagnetically():
    # T = 10.0 >> Tc: tartib buziladi, |m| kichik bo'ladi
    rng = np.random.default_rng(123)
    L = 16
    spins = np.ones((L, L), dtype=int)

    mags = []
    for step in range(150):
        spins = metropolis_sweep(spins, T=10.0, rng=rng)
        if step >= 50:
            mags.append(abs(spins.mean()))

    assert np.mean(mags) < 0.25
