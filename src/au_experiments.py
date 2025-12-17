# au_experiments.py
# Toolkit: robustness sweeps, split ranges, calibrations, Möbius, ablations,
# bootstrap verification, prime-likelihood calibration score, and holdout transfer.

import math, csv, itertools
import numpy as np

# ---------- core helpers ----------
def primes_up_to(m):
    sieve = np.ones(m+1, bool); sieve[:2] = False
    for p in range(2, int(m**0.5)+1):
        if sieve[p]: sieve[p*p::p] = False
    return np.flatnonzero(sieve)

def is_prime_array(nmax):
    sieve = np.ones(nmax+1, bool); sieve[:2] = False
    for p in range(2, int(nmax**0.5)+1):
        if sieve[p]: sieve[p*p::p] = False
    return sieve

def weight(p, mode):
    return math.log(p) if mode=="log" else (1.0/p if mode=="inv" else 1.0)

def fft_conv_same(a, b=None):
    n=a.size; L=1
    while L<2*n: L<<=1
    Fa=np.fft.rfft(a,n=L); Fb=Fa if b is None else np.fft.rfft(b,n=L)
    out=np.fft.irfft(Fa*Fb,n=L)
    return out[:n]

# ---------- curvature (local variance over radius r) ----------
def curvature_local_variance(E, radius=2):
    pad = radius
    Ep = np.pad(E, pad, mode="edge")
    w = 2*radius+1
    shape = (E.size + 2*pad - w + 1, w)
    strides = (Ep.strides[0], Ep.strides[0])
    win = np.lib.stride_tricks.as_strided(Ep, shape=shape, strides=strides)
    local_mean = win.mean(axis=1)
    local_var = ((win - local_mean[:,None])**2).mean(axis=1)
    return local_var

# ---------- interference field on 1..N ----------
def interference_E(N, P=97, weighting="log"):
    small_pr = primes_up_to(P)
    E = np.zeros(N+1, float)
    for p in small_pr:
        E[p::p] += weight(p, weighting)
    return E

# ---------- interference on interval A+1..A+N (split-range) ----------
def interference_E_interval(A, N, P=97, weighting="log"):
    small_pr = primes_up_to(P)
    E_int = np.zeros(N+1, float)  # index 0..N, we use 1..N
    for p in small_pr:
        r = (-A) % p            # first i with (A+i) % p == 0
        i0 = r if r!=0 else p
        for i in range(i0, N+1, p):
            E_int[i] += weight(p, weighting)
    return E_int

# ---------- full analysis (1..N) ----------
def analyze_core(N, P=97, weighting="log", qE=0.10, qK=0.20, curv_radius=2):
    E = interference_E(N, P, weighting)
    K = curvature_local_variance(E, curv_radius)
    T = np.zeros_like(E)
    T[1:-1] = np.abs(E[:-2] - 2*E[1:-1] + E[2:])  # second difference magnitude
    primes = is_prime_array(N).astype(bool)

    qthr_E = float(np.quantile(E[2:], qE))
    qthr_K = float(np.quantile(K[2:], qK))
    quiet = (E <= qthr_E)
    lowK  = (K <= qthr_K)
    joint = quiet & lowK

    mins = (E[1:-1] < E[:-2]) & (E[1:-1] < E[2:])
    min_idx = np.where(mins)[0] + 1

    pr_all  = primes.mean()
    pr_mins = primes[min_idx].mean()
    pr_joint= primes[joint].mean() if np.any(joint) else 0.0
    lift_mins  = pr_mins/pr_all
    lift_joint = pr_joint/pr_all

    return dict(N=N,P=P,weighting=weighting,qE=qE,qK=qK,
                avgE_pr=float(E[primes].mean()),
                avgE_co=float(E[~primes].mean()),
                pr_all=float(pr_all),
                pr_mins=float(pr_mins),
                pr_joint=float(pr_joint),
                lift_mins=float(lift_mins),
                lift_joint=float(lift_joint))

# ---------- parameter sweep ----------
def run_sweep(write_csv=True, csv_name="au_results.csv"):
    combos = list(itertools.product(
        [301,401,601],       # SIZE -> N = SIZE^2
        [97,149,199],        # P
        ["log","inv","one"], # weighting
        [0.10],              # qE
        [0.20,0.30]          # qK
    ))
    rows=[]
    for SIZE,P,W,qE,qK in combos:
        N = SIZE*SIZE
        r = analyze_core(N,P,W,qE,qK)
        r.update(SIZE=SIZE)
        rows.append(r)
        print(f"SIZE={SIZE} P={P} W={W} qK={qK}  lift_joint={r['lift_joint']:.2f}  (lift_Emin={r['lift_mins']:.2f})")
    if write_csv and rows:
        fields = ["SIZE","N","P","weighting","qE","qK","lift_mins","lift_joint","pr_joint","pr_all","avgE_pr","avgE_co"]
        with open(csv_name,"w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
            for r in rows:
                w.writerow({k:r.get(k,"") for k in fields})
        print(f"Wrote {csv_name}")
    return rows

# ---------- split-range validation ----------
def analyze_split_ranges(SIZE=401, P=149, weighting="log", qE=0.10, qK=0.20, curv_radius=2):
    N = SIZE*SIZE
    # Block A: 1..N
    a = analyze_core(N,P,weighting,qE,qK,curv_radius)
    # Block B: N..2N
    E2 = interference_E_interval(N, N, P, weighting)  # 1..N
    K2 = curvature_local_variance(E2, curv_radius)
    primes2 = is_prime_array(2*N).astype(bool)[N:]  # indices N..2N -> length N+1
    qthr_E2 = float(np.quantile(E2[2:], qE))
    qthr_K2 = float(np.quantile(K2[2:], qK))
    quiet2 = (E2 <= qthr_E2)
    lowK2  = (K2 <= qthr_K2)
    joint2 = quiet2 & lowK2
    mins2 = (E2[1:-1] < E2[:-2]) & (E2[1:-1] < E2[2:])
    min_idx2 = np.where(mins2)[0] + 1
    pr_all2  = primes2.mean()
    pr_mins2 = primes2[min_idx2].mean()
    pr_joint2= primes2[joint2].mean() if np.any(joint2) else 0.0
    return dict(
        A_lift_mins=a["lift_mins"], A_lift_joint=a["lift_joint"],
        B_lift_mins=float(pr_mins2/pr_all2), B_lift_joint=float(pr_joint2/pr_all2),
        A_pr_joint=a["pr_joint"], B_pr_joint=float(pr_joint2)
    )

# ---------- simple (old) deciles over all numbers ----------
def calibration_deciles(N=401*401, P=149, weighting="log", curv_radius=2, bins=10):
    E = interference_E(N,P,weighting)
    K = curvature_local_variance(E, curv_radius)
    primes = is_prime_array(N).astype(bool)
    edges_E = np.quantile(E[2:], np.linspace(0,1,bins+1))
    edges_K = np.quantile(K[2:], np.linspace(0,1,bins+1))
    outE=[]; outK=[]
    for i in range(bins):
        maskE = (E >= edges_E[i]) & (E < edges_E[i+1])
        maskK = (K >= edges_K[i]) & (K < edges_K[i+1])
        outE.append((i+1, float(primes[maskE].mean()), float(maskE.mean())))
        outK.append((i+1, float(primes[maskK].mean()), float(maskK.mean())))
    return {"E_bins":outE, "K_bins":outK}

# ---------- improved calibration (fixed) ----------
def calibration_fixed(N=401*401, P=149, weighting="log", qE_quiet=0.20, bins=10):
    """
    Better calibration:
      - Bucket 0: E == 0 (survivors)
      - Buckets 1..bins: deciles of E over E>0 only
      - K-deciles computed *within* quiet E-set (E in lowest qE_quiet)
    Returns dict with 'E_buckets' and 'K_in_quiet'
    """
    E = interference_E(N, P, weighting)
    K = curvature_local_variance(E, radius=2)
    primes = is_prime_array(N).astype(bool)

    # E buckets
    survivors = (E == 0)
    surv_mass = float(survivors.mean())
    surv_prime_rate = float(primes[survivors].mean()) if survivors.any() else float("nan")
    e_buckets = [("E==0", surv_prime_rate, surv_mass)]

    mask_pos = (E > 0)
    if mask_pos.any():
        edges = np.quantile(E[mask_pos], np.linspace(0, 1, bins + 1))
        edges[0] = max(edges[0], np.nextafter(edges[0], float("-inf")))
        for i in range(bins):
            lo, hi = edges[i], edges[i+1]
            m = mask_pos & (E > lo) & (E <= hi)
            mass = float(m.mean())
            prate = float(primes[m].mean()) if m.any() else float("nan")
            e_buckets.append((f"E>0 bin {i+1}", prate, mass))
    else:
        for i in range(bins):
            e_buckets.append((f"E>0 bin {i+1}", float("nan"), 0.0))

    # K deciles inside quiet E
    qthr = float(np.quantile(E[2:], qE_quiet))
    quiet = (E <= qthr)
    k_rows = []
    if quiet.any():
        Kq = K[quiet]
        edgesK = np.quantile(Kq, np.linspace(0, 1, bins + 1))
        edgesK[0] = max(edgesK[0], np.nextafter(edgesK[0], float("-inf")))
        for i in range(bins):
            lo, hi = edgesK[i], edgesK[i+1]
            m = quiet & (K > lo) & (K <= hi)
            mass = float(m.mean())
            prate = float(primes[m].mean()) if m.any() else float("nan")
            k_rows.append((f"K in quiet bin {i+1}", prate, mass))
    else:
        for i in range(bins):
            k_rows.append((f"K in quiet bin {i+1}", float("nan"), 0.0))

    return {"E_buckets": e_buckets, "K_in_quiet": k_rows}

# ---------- Möbius & squarefree comparison ----------
def mobius_array(N):
    mu = np.ones(N+1, int)
    mu[0]=0; mu[1]=1
    spf = np.zeros(N+1, int)
    for i in range(2, N+1):
        if spf[i]==0:
            spf[i]=i
            for j in range(i*i, N+1, i):
                if spf[j]==0: spf[j]=i
    for n in range(2, N+1):
        m=n; prev=0; square=False; cnt=0
        while m>1:
            p = spf[m] if spf[m]!=0 else m
            if p==prev: square=True; break
            prev=p; cnt += 1
            m//=p
            while m>1 and spf[m]==p: square=True; break
            if square: break
        if square: mu[n]=0
        else: mu[n] = -1 if (cnt%2)==1 else 1
    return mu

def compare_mu_vs_E(N=401*401, P=149, weighting="log", qE=0.10):
    E = interference_E(N,P,weighting)
    mu = mobius_array(N)
    primes = is_prime_array(N).astype(bool)
    qthr_E = float(np.quantile(E[2:], qE))
    quiet = (E <= qthr_E)
    frac_sqfree_all   = np.mean(np.abs(mu[2:])==1)
    frac_sqfree_quiet = np.mean((np.abs(mu)==1) & quiet)
    prime_rate_quiet  = np.mean(primes & quiet)
    return dict(frac_sqfree_all=float(frac_sqfree_all),
                frac_sqfree_quiet=float(frac_sqfree_quiet),
                prime_rate_quiet=float(prime_rate_quiet))

# ---------- ablation: shuffle E (should kill lift) ----------
def ablation_shuffle(N=301*301, P=97, weighting="log", qE=0.10, qK=0.20, curv_radius=2, seed=123):
    rng = np.random.default_rng(seed)
    res = analyze_core(N,P,weighting,qE,qK,curv_radius)
    E = interference_E(N,P,weighting)
    rng.shuffle(E[2:])  # keep 0,1 fixed
    K = curvature_local_variance(E, curv_radius)
    primes = is_prime_array(N).astype(bool)
    qthr_E = float(np.quantile(E[2:], qE))
    qthr_K = float(np.quantile(K[2:], qK))
    quiet = (E <= qthr_E); lowK=(K<=qthr_K); joint=quiet&lowK
    mins = (E[1:-1] < E[:-2]) & (E[1:-1] < E[2:])
    min_idx = np.where(mins)[0] + 1
    pr_all  = primes.mean()
    lift_mins_shuf  = (primes[min_idx].mean())/pr_all
    lift_joint_shuf = (primes[joint].mean() if np.any(joint) else 0.0)/pr_all
    return dict(lift_true=res["lift_joint"], lift_shuf=float(lift_joint_shuf),
                liftE_true=res["lift_mins"], liftE_shuf=float(lift_mins_shuf))

# ---------- bootstrap verification on subranges ----------
def _metrics_from_fields(E, K, primes, qE=0.10, qK=0.20):
    """Compute quiet_mass (E==0), quiet_rate, and lift_joint from built arrays."""
    qthr_E = float(np.quantile(E[2:], qE))
    qthr_K = float(np.quantile(K[2:], qK))
    quiet = (E <= qthr_E)
    lowK  = (K <= qthr_K)
    joint = quiet & lowK
    pr_all   = float(primes.mean())
    pr_joint = float(primes[joint].mean()) if np.any(joint) else 0.0
    lift_joint = pr_joint / pr_all if pr_all > 0 else float("nan")
    quiet_mask = (E == 0)
    quiet_mass = float(quiet_mask.mean())
    quiet_rate = float((primes & quiet_mask).mean() / max(quiet_mass, 1e-15))
    return quiet_mass, quiet_rate, lift_joint

def bootstrap_validation(N=401*401, P=149, weighting="log",
                         repeats=20, fraction=0.10, qE=0.10, qK=0.20, seed=42):
    """
    Randomly sample 'repeats' subranges of length L=fraction*N inside [1..N] and
    report mean ± stdev of (quiet_mass, quiet_rate, lift_joint).
    Uses interference_E_interval for fast E on subranges.
    """
    rng = np.random.default_rng(seed)
    L = max(1000, int(fraction*N))
    primes_full = is_prime_array(N).astype(bool)

    quiet_masses, quiet_rates, lifts = [], [], []
    for _ in range(repeats):
        A = rng.integers(0, N - L)  # subrange (A+1 .. A+L)
        E = interference_E_interval(A, L, P=P, weighting=weighting)  # len L+1, idx 0..L
        K = curvature_local_variance(E, radius=2)
        primes = primes_full[A:A+L+1]  # align
        qm, qr, lj = _metrics_from_fields(E, K, primes, qE=qE, qK=qK)
        quiet_masses.append(qm); quiet_rates.append(qr); lifts.append(lj)

    out = dict(
        N=N, P=P, weighting=weighting, repeats=repeats, fraction=fraction,
        quiet_mass_mean=float(np.mean(quiet_masses)), quiet_mass_std=float(np.std(quiet_masses)),
        quiet_rate_mean=float(np.mean(quiet_rates)), quiet_rate_std=float(np.std(quiet_rates)),
        lift_joint_mean=float(np.mean(lifts)), lift_joint_std=float(np.std(lifts)),
    )
    print("\n=== Bootstrap (subranges) ===")
    print(f"N={N:,}  P={P}  weighting={weighting}  repeats={repeats}  fraction={fraction:.2f}")
    print(f"quiet_mass : {out['quiet_mass_mean']:.4f} ± {out['quiet_mass_std']:.4f}")
    print(f"quiet_rate : {out['quiet_rate_mean']:.4f} ± {out['quiet_rate_std']:.4f}")
    print(f"lift_joint : {out['lift_joint_mean']:.2f} ± {out['lift_joint_std']:.2f}")
    return out

# ---------- PRIME-LIKELIHOOD CALIBRATION SCORE (E, K; optional phase via au_phase) ----------
def calibration_score(N=401*401, P=149, weighting="log",
                      use_phase=False, a=1.0, b=1.0, c=0.3, bins=15, order="desc"):
    """
    Build a simple score:
        S(n) = a * 1{E(n)==0}  -  b * K(n)  -  c * C(n)   (C optional, from au_phase.phase_field)
    Then bin S and report prime rate per bin: [(bin_index, prime_rate, mass), ...]
    Set order="desc" (default) to have bin 1 = highest S (should give highest prime rate).
    """
    E = interference_E(N, P, weighting)
    K = curvature_local_variance(E, radius=2)
    primes = is_prime_array(N).astype(bool)

    S = a*(E==0).astype(float) - b*K
    if use_phase:
        try:
            import au_phase as ap
            C = ap.phase_field(N,P,weighting)
            S = S - c*C
        except Exception:
            pass

    edges = np.quantile(S[2:], np.linspace(0,1,bins+1))
    edges[0] = max(edges[0], np.nextafter(edges[0], float("-inf")))

    rows=[]
    if order.lower().startswith("desc"):
        for i in range(bins, 0, -1):
            lo, hi = edges[i-1], edges[i]
            m = (S > lo) & (S <= hi)
            pr = float(primes[m].mean()) if np.any(m) else float("nan")
            mass = float(m.mean())
            rows.append((bins - i + 1, pr, mass))
        rows.sort(key=lambda t: t[0])
    else:
        for i in range(bins):
            lo, hi = edges[i], edges[i+1]
            m = (S > lo) & (S <= hi)
            pr = float(primes[m].mean()) if np.any(m) else float("nan")
            mass = float(m.mean())
            rows.append((i+1, pr, mass))
    return rows

# ---------- Holdout transfer: fit on [1..N], apply to [N..2N] ----------
def score_thresholds_transfer(N=401*401, P=149, weighting="log", bins=15, order="desc"):
    """
    Fit bin thresholds of S = 1{E==0} - K on [1..N], then apply the SAME thresholds to [N..2N].
    Returns list of (bin_index, prime_rate, mass) for the holdout block [N..2N].
    """
    # Fit on [1..N]
    E = interference_E(N, P, weighting)
    K = curvature_local_variance(E, radius=2)
    S = (E==0).astype(float) - K
    edges = np.quantile(S[2:], np.linspace(0,1,bins+1))
    edges[0] = max(edges[0], np.nextafter(edges[0], float("-inf")))

    # Apply to [N..2N]
    E2 = interference_E_interval(N, N, P, weighting)
    K2 = curvature_local_variance(E2, radius=2)
    S2 = (E2==0).astype(float) - K2
    primes2 = is_prime_array(2*N).astype(bool)[N:]  # indices N..2N

    rows=[]
    if order.lower().startswith("desc"):
        for i in range(bins, 0, -1):
            lo, hi = edges[i-1], edges[i]
            m = (S2 > lo) & (S2 <= hi)
            pr = float(primes2[m].mean()) if np.any(m) else float("nan")
            mass = float(m.mean())
            rows.append((bins - i + 1, pr, mass))
        rows.sort(key=lambda t: t[0])
    else:
        for i in range(bins):
            lo, hi = edges[i], edges[i+1]
            m = (S2 > lo) & (S2 <= hi)
            pr = float(primes2[m].mean()) if np.any(m) else float("nan")
            mass = float(m.mean())
            rows.append((i+1, pr, mass))
    return rows

# ---------- Plot helper: prime rate vs score bin ----------
def plot_score_bins(N=401*401, P=149, weighting="log", bins=15, order="desc", save=None):
    import matplotlib.pyplot as plt
    rows = calibration_score(N,P,weighting,use_phase=False,bins=bins,order=order)
    xs = [r[0] for r in rows]; ys = [r[1] for r in rows]
    plt.figure(figsize=(6,4))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Score bin ({}ending)".format("desc" if order.startswith("desc") else "asc"))
    plt.ylabel("Prime rate")
    plt.title(f"Prime rate vs score bin  (N={N:,}, P={P})")
    plt.tight_layout()
    if save:
        import os
        os.makedirs(os.path.dirname(save) or ".", exist_ok=True)
        plt.savefig(save, dpi=160)
        print(f"Saved -> {save}")
    else:
        plt.show()
