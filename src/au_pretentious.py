# au_pretentious.py
# Multiplicative (Euler-product) variants + pretentious distance checks.

import math
import cmath
import numpy as np

# ---------- primes ----------
def primes_up_to(m):
    sieve = np.ones(m+1, dtype=bool)
    sieve[:2] = False
    for p in range(2, int(m**0.5)+1):
        if sieve[p]:
            sieve[p*p::p] = False
    return np.flatnonzero(sieve)

# ---------- per-prime weights and alpha_p ----------
def weight(p, mode):
    if mode == "log": return math.log(p)
    if mode == "inv": return 1.0/p
    if mode == "one": return 1.0
    raise ValueError("weighting must be 'log', 'inv', or 'one'")

def alphas(P=149, weighting="log", scheme="from_weight", c=1.0):
    """
    Returns dict {p: alpha_p} for primes p<=P.
      scheme:
        - 'from_weight' : alpha_p = exp(- w(p))        (your E-based attenuation)
        - 'unit'        : alpha_p = 1                  (this reproduces zeta)
        - 'scaled'      : alpha_p = exp(- c * w(p))    (fit c yourself)
    NOTE: With weighting='log' and scheme='from_weight', alpha_p = 1/p,
          which makes the product resemble ζ(s+1). That's expected.
    """
    ps = primes_up_to(P)
    A = {}
    for p in ps:
        w = weight(p, weighting)
        if scheme == "from_weight":
            A[p] = math.exp(-w)
        elif scheme == "unit":
            A[p] = 1.0
        elif scheme == "scaled":
            A[p] = math.exp(-c * w)
        else:
            raise ValueError("unknown scheme")
    return A

# ---------- truncated Euler product ----------
def Z_truncated(s, alphas_dict):
    """
    Z(s) = ∏_{p≤P} (1 - alpha_p * p^{-s})^{-1}
    (No higher powers included here; you can add them if needed.)
    """
    prod = 1+0j
    for p, a in alphas_dict.items():
        prod *= 1.0 / (1.0 - a * (p ** (-s)))
    return prod

# ---------- sample along critical line sigma=1/2 ----------
def scan_Z_on_critical(P=149, weighting="log",
                       scheme="from_weight", c=1.0,
                       tmax=200.0, nt=2000):
    """
    Returns arrays t, |Z(1/2+it)| for a truncated product.
    """
    A = alphas(P, weighting, scheme, c)
    ts = np.linspace(0.0, tmax, nt)
    mags = []
    for t in ts:
        s = 0.5 + 1j*t
        z = Z_truncated(s, A)
        mags.append(abs(z))
    return ts, np.array(mags)

# ---------- pretentious distance D(f,g; X) ----------
# For multiplicative f, g with values on primes: D^2 = Sum_{p≤X} (1 - Re(f(p) conj g(p)))/p
# We use completely multiplicative f with f(p)=alpha_p (positive real), and g(p)=p^{-it} (zeta-like).
def D_to_n_it(P=149, weighting="log", scheme="from_weight", c=1.0, t=0.0):
    A = alphas(P, weighting, scheme, c)
    s = 0.0
    for p, a in A.items():
        # g(p) = p^{-it} = exp(-i t log p)
        g = cmath.exp(-1j * t * math.log(p))
        s += (1.0 - (a * g).real) / p
    return math.sqrt(max(s, 0.0))

def D_to_zeta_like_best_t(P=149, weighting="log", scheme="from_weight", c=1.0, t_grid=None):
    """
    Minimize D(f, n^{it}; P) over a grid of t. Return (t*, D*).
    """
    if t_grid is None:
        t_grid = np.linspace(-5.0, 5.0, 801)  # coarse is fine
    best_t, best_D = None, float("inf")
    for t in t_grid:
        D = D_to_n_it(P, weighting, scheme, c, t)
        if D < best_D:
            best_D, best_t = D, t
    return best_t, best_D

# ---------- quick report ----------
def quick_report(P_list=(97,149,199), weighting="log"):
    print(f"== Pretentious checks (weighting={weighting}) ==")
    for P in P_list:
        for scheme in ("unit", "from_weight"):
            if scheme=="unit":
                label="ζ baseline (alpha_p=1)"
            else:
                label="from_weight (alpha_p=exp(-w(p)))"
            t_star, D_star = D_to_zeta_like_best_t(P=P, weighting=weighting, scheme=scheme)
            print(f"P={P:3d}  {label:28s}  best t≈{t_star:+.3f},  D≈{D_star:.3f}")
