import numpy as np

# rng = np.random.default_rng(42)
# n0 = 1000
# lam = 0.1
# dt = 0.1
# p = lam * dt
# print("Bir qadamda parchalanish ehtimoli: ", p)
# 
# r = rng.random(n0)
# print("Tasodifiy sonlarning: ", r[:5])
# 
# decayed = r < p
# print("p dan kichik sonlar: ", decayed[:5])
# 
# print("Parchalandi: ", decayed.sum())
# 
# 
# # Bu yerda parchalangan yadrolar qaytib reaksiyada qatnashmaydi.
# 
# n = n0
# print("Boshida: ", n)
# 
# n = n - (rng.random(n) < p).sum()
# print("1-qadamdan keyin: ", n)
# 
# n = n - (rng.random(n) < p).sum()
# print("2-qadamdan keyin: ", n)
# 
# n = n - (rng.random(n) < p).sum()
# print("3-qadamdan keyin: ", n)
# 
# n_steps = 50
# history = np.zeros(n_steps + 1, dtype=int)
# history[0] = n0
# n = n0
# for step in range(n_steps):
#     n = n - (rng.random(n) < p).sum()
#     history[step + 1] = n
# print(history[:6])
# print(history[-1])

def simulate_decay_steps(n0, lam, dt, n_steps, rng):
    p = lam * dt
    history = np.zeros(n_steps + 1, dtype=int)
    history[0] = n0
    n = n0
    for step in range(n_steps):
        n = n - (rng.random(n) < p).sum()
        history[step + 1] = n
    return history


# inverse transform
def sample_decay_times(n, lam, rng):
    u = rng.random(n)
    T = -np.log(1 - u) / lam
    return T


def mle_lambda(times):
    n = len(times)
    lam_hat = n / times.sum()
    se = lam_hat / np.sqrt(n)
    return lam_hat, se


def half_life(times):
    lam_hat, se_lam = mle_lambda(times)
    t_half = np.log(2) / lam_hat
    se_t = t_half / np.sqrt(len(times))
    return t_half, se_t


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    h = simulate_decay_steps(1000, 0.1, 0.1, 50, rng)
    print(h.shape, h[0], h[-1])

    # Nazariya: E[n(k)] = n0 · (1 − p)^k
    runs = 200
    n_steps = 50
    all_h = np.array([simulate_decay_steps(1000, 0.1, 0.1, n_steps, rng) for _ in range(runs)])
    mean_h = all_h.mean(axis=0)
    # nazariy
    k = np.arange(n_steps + 1)
    theory = 1000 * (1 - 0.01) ** k
    print(mean_h[[0, 10, 25, 50]])
    print(theory[[0, 10, 25, 50]])

    rng = np.random.default_rng(42)
    times = sample_decay_times(100_000, 0.1, rng)
    print("Bizda bor vaqt: ", times.shape, times.min() >= 0)
    print("Vaqtlarning o'rtachasi: ", times.mean(), "Nazariy vaqt o'rtachasi: ", 1 / 0.1)
    print("5 sek dan uzoq yashagan yadrolar ulushi: ", (times > 5).mean(), np.exp(-0.1 * 5))

    rng = np.random.default_rng(42)
    times = sample_decay_times(100_000, 0.1, rng)
    lam_hat, se = mle_lambda(times)
    print(lam_hat, se, abs(lam_hat - 0.1))

    t = sample_decay_times(100 * 20_000, 0.1, rng).reshape(20_000, 100)
    lam_hats = 100 / t.sum(axis=1)
    print(lam_hats.std(), 0.1 / np.sqrt(100))

    print("\n=== Yarim yemirilish davri ===\n")

    rng = np.random.default_rng(42)
    times = sample_decay_times(100_000, 0.1, rng)
    t_half, se_t = half_life(times)
    print("Yarim yemirilish davri: ", t_half, ", Standard Error: ", se_t, ", Nazariy: ", np.log(2) / 0.1)
    print("Vaqtning o'rtasidagi qiymat: ", np.median(times))


