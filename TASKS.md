# P0.1 — physics-monte-carlo: Vazifalar

> Qoida: kodni **sen** yozasan. Testlar — bu spetsifikatsiya (nima ishlashi kerakligi). Qanday ishlashini o'zing topasan.
> Qotib qolsang: 1 soat o'zing urin → keyin menga kodingni ko'rsat, hint beraman.

Ishga tushirish:
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -v
```
Boshida barcha testlar **FAIL** bo'ladi — bu normal. Sening vazifang ularni yashil qilish.

---

## 1-hafta

### Task 1 — π ni Monte Carlo bilan baholash → `src/mc/pi.py`
**Funksiya:** `estimate_pi(n: int, rng: np.random.Generator) -> tuple[float, float]`
- Qaytaradi: `(baho, standart_xato)`
- Python `for` loop **ishlatma** — faqat NumPy vectorization
- Standart xatoni ham o'zing chiqar (maslahat: har bir nuqta — Bernoulli tasodifiy kattalik)

**Notebook** `notebooks/01_pi.ipynb`:
- N = 10², 10³, …, 10⁷ uchun haqiqiy xato |baho − π| va standart xatoni log-log grafikda chiz
- Grafik qiyaligini (slope) fit qil. Qancha chiqdi?
- **Savol (notebookda o'z so'zlaring bilan javob ber):** nega xato 1/√N kabi kamayadi? Bu Markaziy limit teoremasiga qanday bog'liq?
- Bonus: Python loop vs NumPy tezligini `%timeit` bilan solishtir

### Task 2 — Radioaktiv parchalanish → `src/mc/decay.py`
**Funksiyalar:**
1. `simulate_decay_steps(n0, lam, dt, n_steps, rng) -> np.ndarray` — har qadamda har bir yadro `lam*dt` ehtimollik bilan parchalanadi. Har qadamdagi qolgan yadrolar sonini qaytaradi (uzunligi `n_steps + 1`, birinchi element `n0`).
2. `sample_decay_times(n, lam, rng) -> np.ndarray` — parchalanish vaqtlarini **inverse transform sampling** usuli bilan generatsiya qil. `rng.exponential` ni **ishlatma** — usulni o'zing chiqar.
3. `mle_lambda(times) -> tuple[float, float]` — λ ning maximum likelihood bahosi va uning xatosi. Formulani qog'ozda o'zing chiqar (likelihood → log → hosila = 0).

**Notebook** `notebooks/02_decay.ipynb`:
- N(t) grafigi + `scipy.optimize.curve_fit` bilan eksponenta fit → yarim yemirilish davri ± xato
- Parchalanish vaqtlari histogrammasi + nazariy egri chiziq ustma-ust
- MLE va curve_fit natijalarini solishtir. Qaysi biri aniqroq? Nega?
- **Savol:** `dt` katta bo'lsa `simulate_decay_steps` nega noto'g'ri natija beradi?

---

## 2-hafta — Task 3: 2D Ising Model va Fazaviy O'tish → `src/mc/ising.py`

> Fizik ma'nosi: ferromagnit modda (masalan, temir) past temperaturada o'z-o'zidan magnitlanadi (spinlar bir tomonga tiziladi), lekin ma'lum kritik temperaturadan ($T_c \approx 2.269$) yuqorida magnit xossasini yo'qotadi. Shu jarayonni **Metropolis algoritmi** bilan noldan simulyatsiya qilasan.

**Funksiyalar (`src/mc/ising.py`):**
1. `lattice_energy(spins: np.ndarray, J: float = 1.0) -> float`
   - `spins` — `(L, L)` o'lchamli, qiymatlari `+1` yoki `-1` bo'lgan 2D massiv.
   - Davriy chegara sharti (periodic boundary conditions): panjaraning oxirgi qatori birinchi qatorga, oxirgi ustuni birinchi ustunga qo'shni (`np.roll` yordam beradi).
   - To'liq energiya: $E = -J \sum_{\langle a,b \rangle} s_a s_b$ (har bir qo'shni juftlik **faqat bir marta** hisoblanadi!).
2. `delta_energy(spins: np.ndarray, i: int, j: int, J: float = 1.0) -> float`
   - Agar faqat `(i, j)` katakdagi spin teskarisiga (`-s[i, j]`) aylantirilsa, energiya qanchaga o'zgaradi ($\Delta E = E_{\text{yangi}} - E_{\text{eski}}$)?
   - Butun panjarani qaytadan hisoblamasdan, faqat 4 ta qo'shni orqali $O(1)$ da hisobla:
     $$\Delta E = 2 J s_{i,j} (s_{i+1,j} + s_{i-1,j} + s_{i,j+1} + s_{i,j-1})$$
     (indekslarda `% L` davriy chegarani unutma).
3. `metropolis_sweep(spins: np.ndarray, T: float, rng: np.random.Generator, J: float = 1.0) -> np.ndarray`
   - Bitta sweep = $L \times L$ marta tasodifiy `(i, j)` spinni tanlab, Metropolis qoidasi bo'yicha aylantirishni sinab ko'rish:
     - Agar $\Delta E \le 0$ bo'lsa — spinni aylantir (`spins[i, j] *= -1`).
     - Agar $\Delta E > 0$ bo'lsa — $\exp(-\Delta E / T)$ ehtimollik bilan aylantir ($k_B = 1$).
   - Yangilangan `spins` massivini qaytaradi.

**Eksperiment (`if __name__ == "__main__":` yoki notebook):**
- $L = 20$ panjarada $T \in [1.0, 3.5]$ oralig'idagi temperaturalar uchun o'rtacha magnitlanish $|M| = |\text{spins.mean()}|$ grafigini chiz va Onsager ning aniq kritik temperaturasi $T_c = 2 / \ln(1 + \sqrt{2}) \approx 2.269$ chizig'ini ko'rsat.
- Bonus: `numba.njit` bilan tezlashtirib, oddiy Python loop bilan tezlik farqini solishtir.

## 3-hafta — Task 4: Z → μμ toy-simulyatsiyasi va invariant massa (keyin beriladi)

---

## Parallel treklar (1-hafta)
- ⚛️ **Fizika (4 soat):** Griffiths, *Introduction to Elementary Particles*, 3-bob (Relativistic Kinematics) — 4-vektorlar, Lorentz o'zgarishlari, invariant massa. Kamida 5 ta masalani qog'ozda yech. 3-haftadagi Task 4 shunga bog'liq.
- 📐 **Matematika (4 soat):** ehtimollar asoslari — Bernoulli, binomial, Puasson, eksponensial taqsimotlar; kutilma va dispersiya; Markaziy limit teoremasi; inverse transform sampling. Har bir tushunchani kichik simulyatsiya bilan tekshir (`notebooks/math/`).
- 📄 **Paper (2 soat):** Metropolis va boshq., 1953 — *"Equation of State Calculations by Fast Computing Machines"*. Bu 2-haftadagi Ising model uchun tayyorgarlik. Konspekt → `papers.md`.
- 🐧 **Linux:** repo'ni `git init` qil, har task uchun alohida commit, GitHub'ga push.

## ✅ 1-hafta tugash mezoni
- [ ] `pytest` — hammasi yashil
- [ ] 2 ta notebook — grafiklar + savollarga javoblar
- [ ] Repo GitHub'da
- [ ] Menga ko'rsatasan → code review
