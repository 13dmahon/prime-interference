# au_plot_scaling.py
# Plots prime enrichment vs interference level across scales

import numpy as np, matplotlib.pyplot as plt
import au_experiments as au

def measure_by_deciles(N, P=149, weighting="log", bins=10):
    """Return (bin centers, prime rate, mass) for E deciles"""
    E = au.interference_E(N, P, weighting)
    primes = au.is_prime_array(N)
    edges = np.quantile(E[2:], np.linspace(0, 1, bins + 1))
    centers = 0.5*(edges[:-1] + edges[1:])
    rates, masses = [], []
    for i in range(bins):
        m = (E >= edges[i]) & (E < edges[i+1])
        if not m.any(): 
            rates.append(np.nan); masses.append(0)
        else:
            rates.append(primes[m].mean()); masses.append(m.mean())
    return centers, np.array(rates), np.array(masses)

def plot_scales(scales=[301,401,601,801], P=149, weighting="log"):
    plt.figure(figsize=(8,5))
    for s in scales:
        N = s*s
        x, y, m = measure_by_deciles(N, P, weighting)
        plt.plot(x, y / np.nanmean(y), marker="o", label=f"N={N:,}")
    plt.xlabel("Interference score E(n)")
    plt.ylabel("Prime rate / mean rate")
    plt.title("Prime enrichment vs interference (normalized)")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_scales()
