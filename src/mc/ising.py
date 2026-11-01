import numpy as np


# 1. FUNKSIYALAR QISMI
def lattice_energy(spins, J=1.0):
    right = np.roll(spins, -1, axis=1)
    down = np.roll(spins, -1, axis=0)
    E = (-J) * np.sum(spins * right + spins * down)
    return float(E)


def delta_energy(spins, i, j, J=1.0):
    L = spins.shape[0]
    s = spins[i, j]
    S = (spins[(i+1) % L, j]
         + spins[(i-1) % L, j]
         + spins[i, (j+1) % L]
         + spins[i, (j-1) % L])
    return float(2 * J * s * S)


def metropolis_sweep(spins, T, rng, J=1.0):
    L = spins.shape[0]
    for _ in range(L * L):
        i = int(rng.integers(0, L))
        j = int(rng.integers(0, L))
        dE = delta_energy(spins, i, j, J=J)
        if dE <= 0 or rng.random() < np.exp(-dE / T):
            spins[i, j] *= -1
    return spins


# 2. SINOV VA EKSPERIMENT QISMI (faqat fayl to'g'ridan-to'g'ri ishga tushirilganda ishlaydi)
if __name__ == "__main__":
    print("--------------------------------------------------------\n"
          "|  Har qanday tizim o‘zining potensial                 |\n"
          "| energiyasini eng kichik (minimal, ya’ni eng manfiy)  |\n"
          "| qiymatga tushirishga intiladi. E<0 barqaror          |\n"
          "--------------------------------------------------------")

    up = np.ones((4, 4))
    print("4*4 panjaradagi energiya: ", lattice_energy(up))

    rng = np.random.default_rng(42)
    rnd = rng.choice([-1, 1], size=(4, 4))
    print("Tasodifiy panjara: ", lattice_energy(rnd))

    board = np.indices((4, 4)).sum(axis=0) % 2 * 2 - 1
    print("Eng yuqori energiyali holat:", lattice_energy(board))

    print(up)
    print("ΔEnergy: ", delta_energy(up, 0, 0))

    spins = rng.choice([-1, 1], size=(4, 4))
    E1 = lattice_energy(spins)
    spins[1, 2] *= -1
    E2 = lattice_energy(spins)
    spins[1, 2] *= -1
    print("Energiyalar farqi: ", E2 - E1, ", ΔEnergy: ", delta_energy(spins, 1, 2))