# au_phase.py
# Computes a simple oscillatory or rotational "phase" field tied to geometry of E.

import numpy as np
from au_experiments import interference_E

def phase_field(N=401*401, P=149, weighting="log", mode="spiral"):
    """
    Compute a simple oscillatory or geometric phase signal C[n].
    - mode="spiral": interpret index as coordinates on a spiral, measure local angle.
    - mode="sin": alternate sinusoidal modulation by index (a simple wave term).
    """
    E = interference_E(N, P, weighting)
    x = np.arange(N+1)

    if mode == "spiral":
        # map each index to coordinates in Ulam-like spiral (approx)
        side = int(np.sqrt(N)) + 1
        X = (x % side) - side/2
        Y = (x // side) - side/2
        angle = np.arctan2(Y, X)
        r = np.sqrt(X**2 + Y**2)
        # damp oscillations by radius, simulate outward swirl
        C = np.sin(6 * angle) * np.exp(-0.0005 * r)
    else:
        # simple sinusoid of index
        C = np.sin(2*np.pi*x/200)

    # normalize to 0..1 range
    C = (C - np.min(C)) / (np.max(C) - np.min(C))
    return C
