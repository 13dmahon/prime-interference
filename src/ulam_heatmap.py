# ulam_heatmap.py
# Upgraded Ulam view: show primes on a spiral and color by interference E(n),
# then print stats (including pair gaps).

import math
import numpy as np
import matplotlib.pyplot as plt

# ===================== Parameters =====================
SIZE       = 301      # spiral grid side (odd). Total N = SIZE^2
P          = 97       # include small primes up to P in E(n)
WEIGHTING  = "log"    # "log" -> log(p); "inv" -> 1/p; "one" -> 1
SHOW_LINKS = False    # sketch a few multiplicative links p -> pq
OVERLAY_QUIET = True  # lightly tint quietest 10% underneath primes
CMAP = "inferno"      # try "inferno_r" to invert, or "viridis"
SAVE_PNG = True
PNG_NAME = "ulam_heatmap.png"
# ======================================================

# ---------- utilities ----------
def primes_up_to(m: int) -> np.ndarray:
    sieve = np.ones(m + 1, dtype=bool)
    sieve[:2] = False
    for p in range(2, int(m ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p :: p] = False
    return np.flatnonzero(sieve)

def is_prime_array(nmax: int) -> np.ndarray:
    sieve = np.ones(nmax + 1, dtype=bool)
    sieve[:2] = False
    for p in range(2, int(nmax ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p :: p] = False
    return sieve

def ulam_spiral(size: int) -> np.ndarray:
    n = size * size
    grid = np.zeros((size, size), dtype=int)
    x = y = size // 2
    dx, dy = 0, -1
    for v in range(1, n + 1):
        if 0 <= x < size and 0 <= y < size:
            grid[y, x] = v
        # turn condition
        if x == y or (x < y and x + y == size - 1) or (x > y and x + y == size):
            dx, dy = -dy, dx
        x += dx
        y += dy
    return grid

# ---------- build spiral ----------
assert SIZE % 2 == 1, "SIZE should be odd (301, 401, ...)"
grid = ulam_spiral(SIZE)
N = SIZE * SIZE

# ---------- interference score E(n) ----------
small_pr = primes_up_to(P)
E = np.zeros(N + 1, dtype=float)

def weight(p: int) -> float:
    if WEIGHTING == "log":
        return math.log(p)
    if WEIGHTING == "inv":
        return 1.0 / p
    return 1.0

for p in small_pr:
    w = weight(p)
    E[p :: p] += w   # add weight to multiples of p

# ---------- primality flags ----------
prime_flag = is_prime_array(N)             # boolean array length N+1

# ---------- heatmap data ----------
E_grid = np.zeros_like(grid, dtype=float)
for y in range(SIZE):
    for x in range(SIZE):
        E_grid[y, x] = E[grid[y, x]]

# ---------- plot ----------
plt.figure(figsize=(8, 8))
im = plt.imshow(E_grid, cmap=CMAP, interpolation="nearest")
plt.colorbar(im, label="Interference score E(n)")
plt.title("Interference heatmap on Ulam spiral (dark = quiet)")

# overlay primes
py, px = np.where(prime_flag[grid])
plt.scatter(px, py, s=2, c="cyan", alpha=0.7, label="Primes")

# optional multiplicative links
if SHOW_LINKS:
    coords = {}
    for y in range(SIZE):
        for x in range(SIZE):
            coords[grid[y, x]] = (x, y)
    drawn = 0
    for p in small_pr[:10]:
        for q in small_pr[:10]:
            pq = p * q
            if pq <= N and p in coords and pq in coords:
                x1, y1 = coords[p]
                x2, y2 = coords[pq]
                plt.plot([x1, x2], [y1, y2], linewidth=0.4, alpha=0.4, color="white")
                drawn += 1
                if drawn > 300:
                    break

# optional quiet overlay
if OVERLAY_QUIET:
    q10 = float(np.quantile(E[2:], 0.10))  # ignore 0,1
    mask_quiet_grid = E_grid <= q10
    plt.imshow(mask_quiet_grid, alpha=0.12, cmap="Greens")

plt.axis("off")
plt.legend(loc="upper left", markerscale=4)
plt.tight_layout()

# ---------- stats ----------
prime_1d = prime_flag.astype(bool)

# averages
avg_E_primes     = float(E[prime_1d].mean())
avg_E_composites = float(E[~prime_1d].mean())

# quietest 10% enrichment
q10 = float(np.quantile(E[2:], 0.10))
quiet_mask_1d = (E <= q10)
quiet_prime_count = int(np.count_nonzero(prime_1d & quiet_mask_1d))
prime_total       = int(np.count_nonzero(prime_1d))
quiet_pct = quiet_prime_count / prime_total if prime_total else 0.0

# local minima enrichment
mins = (E[1:-1] < E[0:-2]) & (E[1:-1] < E[2:])
min_idx = np.where(mins)[0] + 1
prime_rate_all  = prime_1d.mean()
prime_rate_mins = prime_1d[min_idx].mean() if len(min_idx) else 0.0
lift = (prime_rate_mins / prime_rate_all) if prime_rate_all else float("nan")

print("\n--- Interference stats ---")
print(f"N = {N:,}, P = {P}, weighting = {WEIGHTING}")
print(f"Average E(primes):      {avg_E_primes:.6f}")
print(f"Average E(composites):  {avg_E_composites:.6f}")
print(f"Primes in quietest 10%: {quiet_prime_count}/{prime_total} ({quiet_pct:.2%})")
print(f"Prime rate overall:     {prime_rate_all:.4f}")
print(f"Prime rate at minima:   {prime_rate_mins:.4f}")
print(f"Lift (minima/overall):  {lift:.2f}x")

# ---------- prime-pair interference ----------
def pair_stats(gap: int):
    valid = np.arange(2, N - gap + 1)
    pair_flag = prime_1d[valid] & prime_1d[valid + gap]
    Eg = E[valid] + E[valid + gap]  # pair interference

    total_pairs = int(np.count_nonzero(pair_flag))
    print(f"\nPrime-pair interference for gap {gap}:")
    if total_pairs == 0:
        print("  (no pairs in range)")
        return

    for q in (0.10, 0.20, 0.30):
        thr = float(np.quantile(Eg, q))
        in_quiet = Eg <= thr
        pairs_in_quiet = int(np.count_nonzero(pair_flag & in_quiet))
        pct = pairs_in_quiet / total_pairs
        print(f"  quietest {int(q*100):>2}%: {pairs_in_quiet}/{total_pairs} = {pct:.2%}")

pair_stats(2)   # twin primes
pair_stats(6)   # sexy primes

# ---------- save & show ----------
if SAVE_PNG:
    plt.savefig(PNG_NAME, dpi=180)
plt.show()
