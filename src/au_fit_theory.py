# au_fit_theory.py
# Empirical prime law vs sieve prediction + scaling table + exports

import os, math, csv
import numpy as np, matplotlib.pyplot as plt
import au_experiments as au

# ---- helpers from earlier ----
def small_primes(P): return au.primes_up_to(P)

def phi_P(P):
    prod = 1.0
    for p in small_primes(P): prod *= (1.0 - 1.0/p)
    return prod

def mean_one_over_log_on_range(N, A=2):
    n = np.arange(max(A,2), N+1, dtype=float)
    return float(np.mean(1.0/np.log(n)))

def prime_rate_by_E_bins(N, P=149, weighting="log", bins=20):
    E = au.interference_E(N, P, weighting)
    primes = au.is_prime_array(N).astype(bool)
    mask0 = (E == 0)
    E0_rate = float(primes[mask0].mean()) if mask0.any() else float("nan")
    E0_mass = float(mask0.mean())
    mask_pos = (E > 0)
    if not mask_pos.any():
        return np.array([]), np.array([]), np.array([]), E0_rate, E0_mass
    edges = np.quantile(E[mask_pos], np.linspace(0,1,bins+1))
    edges[0] = max(edges[0], np.nextafter(edges[0], float("-inf")))
    centers, rates, masses = [], [], []
    for i in range(bins):
        lo, hi = edges[i], edges[i+1]
        m = mask_pos & (E > lo) & (E <= hi)
        centers.append(0.5*(lo+hi))
        if m.any():
            rates.append(float(primes[m].mean()))
            masses.append(float(m.mean()))
        else:
            rates.append(float("nan")); masses.append(0.0)
    return np.array(centers), np.array(rates), np.array(masses), E0_rate, E0_mass

def fit_exp_to_bins(E_centers, rates):
    mask = np.isfinite(rates) & (rates > 0)
    if mask.sum() < 2: return float("nan"), float("nan"), 0
    x = E_centers[mask]; y = np.log(rates[mask])
    b1, b0 = np.polyfit(x, y, 1)
    return math.exp(b0), -b1, mask.sum()

# ---- main analysis + plot ----
def run_fit_and_plot(N=401*401, P=149, weighting="log", bins=20,
                     show=True, save_dir=None, tag=None):
    E_centers, E_rates, E_mass, E0_rate, E0_mass = prime_rate_by_E_bins(N,P,weighting,bins)
    # (optional) fit only the small-E tail to avoid noise
    mask_smallE = (np.isfinite(E_centers)) & (E_centers > 0) & (E_centers <= 2.5)
    A, alpha, used = fit_exp_to_bins(E_centers[mask_smallE], E_rates[mask_smallE])

    phi = phi_P(P)
    mean_invlog = mean_one_over_log_on_range(N)
    sieve_quiet_rate = mean_invlog / phi

    print("\n=== Empirical prime law vs sieve prediction ===")
    print(f"N = {N:,}, P = {P}, weighting = {weighting}, bins = {bins}")
    print(f"Measured quiet mass    (E==0): {E0_mass:.4f}")
    print(f"Sieve φ_P (survivors):        {phi:.4f}")
    print(f"Measured quiet prime rate:    {E0_rate:.4f}")
    print(f"Heuristic quiet prime rate:   {sieve_quiet_rate:.4f}")
    print(f"Exp fit on small E>0:  rate(E) ≈ {A:.4e} * exp(-{alpha:.4f} * E)  using {used} bins")

    # plot
    plt.figure(figsize=(8,5))
    plt.scatter(E_centers, E_rates, s=30, label="Measured rate per E-bin")
    if np.isfinite(A) and np.isfinite(alpha) and used>0:
        xs = np.linspace(max(1e-6, np.nanmin(E_centers)), np.nanmax(E_centers), 200)
        plt.plot(xs, A*np.exp(-alpha*xs), label="Exp fit A·e^{-αE}")
    plt.scatter([0.0], [E0_rate], s=70, label="E==0 (quiet)", marker="o")
    plt.axhline(mean_invlog, linestyle="--", label="Baseline ~ average 1/log n")
    plt.axhline(sieve_quiet_rate, linestyle=":", label="Heuristic quiet rate ~ (1/log)/φ_P")
    plt.yscale("log")
    plt.xlabel("Interference E(n)"); plt.ylabel("Prime rate (log scale)")
    ttl = f"Prime rate vs interference E  (N={N:,}, P={P})"
    if tag: ttl += f"  [{tag}]"
    plt.title(ttl); plt.legend(); plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fn = os.path.join(save_dir, f"prime_rate_vs_E_N{N}_P{P}{'_'+tag if tag else ''}.png")
        plt.savefig(fn, dpi=180)
        print(f"Saved plot -> {fn}")
    if show: plt.show()

    # return row
    return dict(N=N,P=P,weighting=weighting,bins=bins,
                quiet_mass=E0_mass, quiet_rate=E0_rate,
                sieve_phi=phi, mean_invlog=mean_invlog,
                sieve_quiet_rate=sieve_quiet_rate,
                A=A, alpha=alpha, used_bins=used)

def scaling_table(SIZES=(301,401,601,801), P=149, weighting="log",
                  save_csv=None, save_dir=None):
    rows=[]
    for s in SIZES:
        N = s*s
        r = run_fit_and_plot(N, P, weighting, bins=20, show=False, save_dir=save_dir)
        core = au.analyze_core(N, P, weighting)
        r.update(lift_joint=core["lift_joint"], lift_mins=core["lift_mins"], pr_all=core["pr_all"])
        rows.append(r)

    print("\n=== Scaling summary ===")
    header = f"{'N':>10}  {'quiet_mass':>10}  {'quiet_rate':>10}  {'φ_P':>8}  {'(1/log)/φ_P':>12}  {'lift_joint':>10}  {'alpha':>8}"
    print(header)
    for r in rows:
        print(f"{r['N']:>10,}  {r['quiet_mass']:>10.4f}  {r['quiet_rate']:>10.4f}  {r['sieve_phi']:>8.4f}  {r['sieve_quiet_rate']:>12.4f}  {r['lift_joint']:>10.2f}  {r['alpha']:>8.3f}")

    if save_csv:
        os.makedirs(os.path.dirname(save_csv) or ".", exist_ok=True)
        with open(save_csv,"w",newline="") as f:
            w=csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"Wrote CSV -> {save_csv}")
    return rows

if __name__ == "__main__":
    # single run (shows figure)
    run_fit_and_plot(N=401*401, P=149, weighting="log", bins=20, show=True, save_dir=None)
    # scaling bundle (uncomment to export)
    # scaling_table(SIZES=(301,401,601,801), P=149, save_csv="au_scaling.csv", save_dir="figs_scaling")
