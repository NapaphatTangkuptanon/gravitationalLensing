# task7_ascii.py  — Parameter study without Unicode in identifiers/strings (Windows-safe)

import os, json, math
import numpy as np
import matplotlib.pyplot as plt

# -------------------
# Grid / geometry
# -------------------
L = 6.0
N = 512
dth = L / N
th = np.linspace(-L/2, L/2, N)
thx, thy = np.meshgrid(th, th, indexing="xy")
extent = [th.min(), th.max(), th.min(), th.max()]

# Optional SciPy features (graceful fallback if not installed)
try:
    from scipy.ndimage import gaussian_filter, label, center_of_mass, maximum_filter
except Exception:
    gaussian_filter = None
    label = None
    center_of_mass = None
    maximum_filter = None

# -------------------
# Study parameters
# -------------------
GAMMAS = [0.0, 0.05, 0.1, 0.2]
BETA_OFFSETS = [0.0, 0.1, 0.2]
PHI_GAMMA_DEG = 30.0

# -------------------
# Source (Sersic) params
# -------------------
S_n   = 1.0
S_Re  = 0.25
S_q   = 0.7
S_phi = -20.0
S_I0  = 5000.0

# -------------------
# Models & helpers
# -------------------
def sersic_intensity(beta_x, beta_y, n, Re, q, phi_deg, beta0, I0):
    """Elliptical Sersic profile centered at beta0."""
    phi = np.deg2rad(phi_deg)
    c, s = np.cos(phi), np.sin(phi)
    bx = beta_x - beta0[0]
    by = beta_y - beta0[1]
    xr =  c*bx + s*by
    yr = -s*bx + c*by
    r_ell = np.sqrt(xr*xr + (yr/q)*(yr/q))
    b_n = 2*n - 1/3 + 4/(405*n) + 46/(25515*n*n)
    return I0 * np.exp(-b_n*((r_ell/Re)**(1/n) - 1.0))

def psi_shear(thx, thy, gamma, phi_deg):
    """External shear potential."""
    if gamma == 0.0:
        return np.zeros_like(thx)
    phi = np.deg2rad(phi_deg)
    c2, s2 = np.cos(2*phi), np.sin(2*phi)
    return 0.5*gamma*((thx**2 - thy**2)*c2 + 2*thx*thy*s2)

def estimate_ring_radius(I, top_frac=0.01, smooth_sigma=1.0):
    """Estimate ring radius from the brightest top_frac of pixels."""
    if I is None:
        return float("nan"), float("nan")
    if gaussian_filter is not None and smooth_sigma > 0:
        Is = gaussian_filter(I, smooth_sigma)
    else:
        Is = I
    thr = np.quantile(Is, 1.0 - top_frac)
    mask = Is >= thr
    r = np.hypot(thx[mask], thy[mask])
    if r.size == 0:
        return float("nan"), float("nan")
    return float(np.median(r)), float(np.std(r))

def morphology_tag(I):
    """Rough morphology: 'ring', 'arcs', or 'single'."""
    if I is None:
        return "single"
    r_med, r_std = estimate_ring_radius(I)
    if not math.isnan(r_med) and r_std < 0.15*abs(r_med):
        return "ring"

    thr = np.quantile(I, 0.995)
    mask = I >= thr
    if label is not None and center_of_mass is not None:
        lab, n = label(mask)
        n_big = 0
        for k in range(1, n+1):
            if (lab == k).sum() >= 5:
                n_big += 1
        return "arcs" if n_big >= 2 else "single"

    if maximum_filter is not None:
        peaks = (I == maximum_filter(I, size=5)) & mask
        n_peaks = int(peaks.sum())
        return "arcs" if n_peaks >= 2 else "single"

    return "single"

def simulate_once(psi_base, gamma, beta0):
    """Compose psi_total = psi_base + psi_shear; ray-trace; measure."""
    psi_tot = psi_base + psi_shear(thx, thy, gamma, PHI_GAMMA_DEG)

    # alpha = grad psi
    psi_x, psi_y = np.gradient(psi_tot, dth, edge_order=2)
    alpha_x, alpha_y = psi_x, psi_y

    # beta(theta) = theta - alpha(theta)
    beta_x = thx - alpha_x
    beta_y = thy - alpha_y

    # Unlensed source sample (same grid, just for normalization)
    S_beta = sersic_intensity(thx, thy, S_n, S_Re, S_q, S_phi, beta0, S_I0)

    # Ray-traced lensed image
    I_ray = sersic_intensity(beta_x, beta_y, S_n, S_Re, S_q, S_phi, beta0, S_I0)

    # Measurements
    r_med, r_std = estimate_ring_radius(I_ray, top_frac=0.01, smooth_sigma=1.0)
    M = float(I_ray.sum() / S_beta.sum()) if S_beta.sum() > 0 else float("nan")
    morph = morphology_tag(I_ray)
    return r_med, r_std, M, morph, I_ray

# -------------------
# Main
# -------------------
def main():
    psi_f = "psi.npy"
    if not os.path.exists(psi_f):
        raise FileNotFoundError("Missing psi.npy (run Task 1 first to save it).")
    psi_base = np.load(psi_f)

    rows = []
    thumbs = []

    for gamma in GAMMAS:
        for bmag in BETA_OFFSETS:
            beta0 = (bmag, 0.0)  # offset along +x (arcsec)
            r_med, r_std, M, morph, I_ray = simulate_once(psi_base, gamma, beta0)
            rows.append(dict(
                gamma=float(gamma),
                beta0_mag=float(bmag),
                ring_radius_arcsec=float(r_med),
                ring_width_std=float(r_std),
                magnification=float(M),
                morphology=str(morph)
            ))
            if len(thumbs) < 6:
                thumbs.append((gamma, bmag, I_ray))

    # Console report (ASCII-only header)
    header = f"{'gamma':>6} | {'beta0':>6} | {'ring_r':>8} | {'sigma_r':>7} | {'M':>7} | morphology"
    print("\n=== Task 7: Parameter Study (gamma, |beta0|) ===")
    print(header); print("-"*len(header))
    for r in rows:
        print(f"{r['gamma']:6.2f} | {r['beta0_mag']:6.2f} | {r['ring_radius_arcsec']:8.3f} | "
              f"{r['ring_width_std']:7.3f} | {r['magnification']:7.3f} | {r['morphology']}")

    # Save table (force UTF-8 to be safe on Windows)
    with open("task7_table.txt", "w", encoding="utf-8") as f:
        f.write(header + "\n")
        f.write("-"*len(header) + "\n")
        for r in rows:
            f.write(f"{r['gamma']:6.2f} | {r['beta0_mag']:6.2f} | {r['ring_radius_arcsec']:8.3f} | "
                    f"{r['ring_width_std']:7.3f} | {r['magnification']:7.3f} | {r['morphology']}\n")

    # Save JSON (keep unicode allowed just in case, but all keys/values here are ASCII)
    with open("task7_results.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print("Wrote: task7_table.txt and task7_results.json")

    # Thumbnail grid of first few cases
    if len(thumbs) > 0:
        n = len(thumbs)
        cols = 3
        rows_fig = (n + cols - 1) // cols
        plt.figure(figsize=(4*cols, 4*rows_fig))
        for i, (g, b, Im) in enumerate(thumbs, start=1):
            plt.subplot(rows_fig, cols, i)
            plt.imshow(Im, origin="lower", extent=extent, cmap="magma")
            plt.title(f"gamma={g:.2f}, |beta0|={b:.2f}")
            plt.xticks([]); plt.yticks([])
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
