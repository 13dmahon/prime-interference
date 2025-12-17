# au_heuristic.py
# Prime-clustering heuristic from interference E (and optional curvature K).

import math
import csv
import numpy as np

# Reuse your au_experiments helpers
import au_experiments as au

# ---------- fit Pr(prime | E) ≈ A * exp(-alpha * E) ----------
def fit_prime_law(N, P=149, weighting="log", bins=20, eps=1e-12):
    """
    Bin by E>0, estimate prime-rate per bin, fit log(rate) = log A - alpha * E (OLS on small E region).
    Sets A so that the global mean matches empirical base rate (stabilizes fit).
    Returns dict with A, alpha, base_rate, phi_P, quiet_mass, quiet_rate.
    """
    E = au.interference_E(N, P, weighting)
    primes = au.is_prime_array(N).astype(bool)

    # Quiet mass & phi_P check
    quiet_mass = float((E == 0).mean())
    # φ_P (sieve survivors): ∏_{p≤P}(1 - 1/p)
    ps = au.primes_up_to(P)
    phi_P = 1.0
    for p in ps: phi_P *= (1.0 - 1.0/p)
    quiet_rate = float((primes & (E==0)).mean() / max(quiet_mass, eps))
    base_rate = float(primes.mean())

    # E>0 bins
    mask_pos = (E > 0)
    if not np.any(mask_pos):
        return dict(A=np.nan, alpha=np.nan, base_rate=base_rate,
                    phi_P=phi_P, quiet_mass=quiet_mass, quiet_rate=quiet_rate)

    edges = np.quantile(E[mask_pos], np.linspace(0,1,bins+1))
    edges[0] = max(edges[0], np.nextafter(edges[0], float("-inf")))
    Eb, Rb, Wb = [], [], []
    for i in range(min(10, bins)):  # fit only the lowest 10 bins (small E region)
        lo, hi = edges[i], edges[i+1]
        m = mask_pos & (E > lo) & (E <= hi)
        if not np.any(m): continue
        rate = float(primes[m].mean())
        if rate <= 0: continue
        Eb.append(float(E[m].mean()))
        Rb.append(math.log(rate))
        Wb.append(float(m.sum()))
    if len(Eb) < 2:
        return dict(A=np.nan, alpha=np.nan, base_rate=base_rate,
                    phi_P=phi_P, quiet_mass=quiet_mass, quiet_rate=quiet_rate)

    # Weighted least squares for log rate ~ a0 - alpha * E
    X = np.vstack([np.ones(len(Eb)), -np.array(Eb)]).T
    y = np.array(Rb)
    w = np.array(Wb)
    XtWX = X.T @ (w[:,None]*X)
    XtWy = X.T @ (w*y)
    beta = np.linalg.solve(XtWX, XtWy)  # [a0, alpha]
    a0, alpha = beta[0], beta[1]
    A = math.exp(a0)

    # Stabilize A to match base rate on average: rescale so mean predicted = base_rate
    pred = predict_prob_from_E(E, A, alpha)
    scale = base_rate / max(pred.mean(), eps)
    A *= scale

    return dict(A=A, alpha=alpha, base_rate=base_rate,
                phi_P=phi_P, quiet_mass=quiet_mass, quiet_rate=quiet_rate)

def predict_prob_from_E(E, A, alpha):
    Pn = A * np.exp(-alpha * E)
    # clamp to [0, 1]
    return np.clip(Pn, 0.0, 1.0)

# Optional mix with curvature: S = 1{E==0} - K, turn into prob by monotone rescaling
def mixed_probability(E, K, A, alpha, w_s=0.0):
    """
    If w_s=0: pure E-law. If w_s>0: add a monotone bump from score S to break ties inside quiet set.
    """
    base = predict_prob_from_E(E, A, alpha)
    if w_s <= 0: return base
    S = (E==0).astype(float) - K
    # rank-normalize S to [0,1] to keep monotonicity
    idx = np.argsort(S, kind="mergesort")
    ranks = np.empty_like(idx, dtype=float); ranks[idx] = np.linspace(0,1,len(S))
    return np.clip((1-w_s)*base + w_s * ranks, 0.0, 1.0)

# ---------- evaluation: lift curve, precision@k, AUC ----------
def eval_metrics(prob, primes, ks=(100, 500, 1000, 5000)):
    """
    prob: array of predicted probabilities (len N+1).
    primes: boolean primality mask (len N+1).
    Returns dict with base_rate, precision@k, lift@k, and AUC (rank-based).
    """
    N = len(prob)-1
    pr = primes.astype(bool)
    base = float(pr.mean())

    # sort by prob desc (exclude 0 and 1 to be safe)
    order = np.argsort(prob[2:])[::-1] + 2
    y = pr[order].astype(int)

    # precision@k and lift@k
    prec_k = {}
    lift_k = {}
    for k in ks:
        k = min(k, len(order))
        p = y[:k].mean() if k>0 else float("nan")
        prec_k[k] = float(p)
        lift_k[k] = float(p/base) if base>0 else float("nan")

    # AUC via rank method
    # AUC = (sum of ranks of positives - n_pos*(n_pos+1)/2) / (n_pos*n_neg)
    ranks = np.empty_like(order, dtype=float)
    ranks[np.argsort(order)] = np.arange(1, len(order)+1)  # actual positions irrelevant; we need ranks over prob
    # But easier: use the prob order's ranks directly
    # Build ranks for y (positives among sorted)
    pos_idx = np.where(y==1)[0]
    n_pos = len(pos_idx); n_neg = len(y) - n_pos
    if n_pos==0 or n_neg==0:
        auc = float("nan")
    else:
        auc = (np.sum(pos_idx+1) - n_pos*(n_pos+1)/2) / (n_pos*n_neg)
    return dict(base_rate=base, precision_at_k=prec_k, lift_at_k=lift_k, auc=float(auc))

# ---------- train on [1..N], test on [N..2N], export top candidates ----------
def train_and_holdout(N, P=149, weighting="log", w_s=0.0, top_m=2000, csv_path="prime_candidates.csv"):
    # Train on [1..N]
    fit = fit_prime_law(N, P, weighting)
    A, alpha = fit["A"], fit["alpha"]
    if not np.isfinite(A) or not np.isfinite(alpha):
        print("Fit failed."); return dict(ok=False, fit=fit)

    E_A = au.interference_E(N, P, weighting)
    K_A = au.curvature_local_variance(E_A, radius=2)
    prob_A = mixed_probability(E_A, K_A, A, alpha, w_s=w_s)
    primes_A = au.is_prime_array(N)
    metrics_A = eval_metrics(prob_A, primes_A)

    # Holdout [N..2N]
    E_B = au.interference_E_interval(N, N, P, weighting)
    K_B = au.curvature_local_variance(E_B, radius=2)
    prob_B = mixed_probability(E_B, K_B, A, alpha, w_s=w_s)
    primes_2N = au.is_prime_array(2*N)
    primes_B = primes_2N[N:]  # align to E_B length

    metrics_B = eval_metrics(prob_B, primes_B)

    # Export top candidates from holdout
    order_B = np.argsort(prob_B[2:])[::-1] + 2  # best first
    top = order_B[:top_m]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "prob", "E", "K"])
        for n in top:
            w.writerow([int(n), float(prob_B[n]), float(E_B[n]), float(K_B[n])])

    out = dict(ok=True, fit=fit, metrics_train=metrics_A, metrics_holdout=metrics_B, csv=csv_path)
    # Pretty print
    print("\n=== Heuristic fit ===")
    print(f"N={N:,}  P={P}  weighting={weighting}  ->  A={A:.6g}  alpha={alpha:.3f}")
    print(f"quiet_mass≈φ_P: {fit['quiet_mass']:.4f} vs {fit['phi_P']:.4f} | quiet_rate≈ {(fit['quiet_rate']):.4f}")
    print("\nTrain [1..N]:")
    print(f" base_rate={metrics_A['base_rate']:.4f}  AUC={metrics_A['auc']:.3f}")
    for k,v in metrics_A['lift_at_k'].items(): print(f"  lift@{k}: {v:.2f}x")
    print("\nHoldout [N..2N]:")
    print(f" base_rate={metrics_B['base_rate']:.4f}  AUC={metrics_B['auc']:.3f}")
    for k,v in metrics_B['lift_at_k'].items(): print(f"  lift@{k}: {v:.2f}x")
    print(f"\nExported top {top_m} holdout candidates -> {csv_path}")
    return out
