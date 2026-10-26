from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1-QISM: FUNKSIYALAR
# Bu qism `from mc.pi import estimate_pi` qilinganda yuklanadi.
# Bu yerda faqat funksiya ta'riflari bor — hech narsa ishga tushmaydi.
# ============================================================

def estimate_pi(n, rng):
    x = rng.uniform(-1, 1, size=n)
    y = rng.uniform(-1, 1, size=n)
    d2 = x**2 + y**2
    inside = d2 <= 1
    p_hat = inside.mean()
    baho = 4 * p_hat
    se = 4 * np.sqrt(p_hat * (1 - p_hat) / n)
    return baho, se


def estimate_pi_many(n, R, rng):
    x = rng.uniform(-1, 1, size=(R, n))
    y = rng.uniform(-1, 1, size=(R, n))
    d2 = x**2 + y**2
    inside = d2 <= 1
    return 4 * (inside.sum(axis=1) / n)


# ============================================================
# 2-QISM: EKSPERIMENTLAR
# Bu blok FAQAT `python pi.py` deb to'g'ridan-to'g'ri ishga tushirilganda bajariladi.
# Import qilinganda (testlar, notebook, boshqa modullar) — bajarilmaydi.
# ============================================================

if __name__ == "__main__":
    N = 1000000
    rng = np.random.default_rng(42)

    x = rng.uniform(-1, 1, size=N)
    y = rng.uniform(-1, 1, size=N)

    print(x.shape, y.shape)
    print(x.min(), x.max())

    d2 = x**2 + y**2
    print(d2.shape)
    print(d2.min(), d2.max())

    inside = d2 <= 1
    print(inside.shape, inside.dtype)
    print(inside[:10])
    print(inside.sum())
    print("\n*****************************\n")
    I = inside.astype(float)
    print(I[:10])
    print(I.mean())
    # dispersiyani hisoblash
    print("\n=== Bernulli natijasining dispersiyasi ===")
    p_hat = I.mean()
    var_direct = ((I - p_hat) ** 2).mean()
    print("p =", p_hat)
    print(var_direct)
    print(I.var())      # numpy ning tayyor funksiyasi
    print(p_hat*(1-p_hat))

    print("\n=== pi ni hisoblash ===")
    pi_est = 4 * (inside.sum() / N)
    print(pi_est)
    print(np.pi)
    print(abs(pi_est - np.pi))

    # N ga bog'liqlik

    print("\n=== N ga bog'liqlik ===\n")

    baho, se = estimate_pi(100, rng)
    print(baho, se, abs(baho - np.pi))

    baho, se = estimate_pi(10_000, rng)
    print(baho, se, abs(baho - np.pi))

    baho, se = estimate_pi(1_000_000, rng)
    print(baho, se, abs(baho - np.pi))

    print("\n=== 2D massivga joylash ===\n")

    est = estimate_pi_many(100, 200, rng)
    print(est.shape)
    print(est.mean())
    print(est.std())

    print("\n=== Turli N lar uchun o'rtacha xatolik ===\n")

    Ns = np.array([10**2, 10**3, 10**4, 10**5])
    R = 100
    errors = np.array([np.abs(estimate_pi_many(n, R, rng) - np.pi).mean() for n in Ns])
    print(errors)
    print(errors[:-1] / errors[1:])

    print("\n=== SE tekshiruvi ===")
    print((I**2 == I).all())

    n = 100
    se_theory = 4 * np.sqrt(p_hat * (1 - p_hat) / n)
    print("nazariy SE:", se_theory)

    est_big = estimate_pi_many(n, 20_000, rng)
    print("haqiqiy tebranish:", est_big.std())

    # Grafik chizish

    plt.loglog(Ns, errors, "o-", label="hisoblangan xatolik")
    plt.loglog(Ns, errors[0] * np.sqrt(Ns[0] / Ns), "--", label="1/√N (nazariy)")
    plt.xlabel("N (nuqtalar soni)")
    plt.ylabel("o'rtacha absolyut xatolik")
    plt.legend()
    plt.grid(True)
    # Yo'l shu faylning joylashuviga nisbatan — qaysi papkadan ishga tushirilishidan qat'i nazar ishlaydi
    out_path = Path(__file__).resolve().parents[2] / "notebooks" / "01_pi.png"
    out_path.parent.mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=150)

    slope = np.polyfit(np.log10(Ns), np.log10(errors), 1)[0]
    print(slope)