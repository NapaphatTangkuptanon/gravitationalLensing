import os, json
import numpy as np
import matplotlib.pyplot as plt

L = 6.0
N = 512
dth = L / N
th = np.linspace(-L/2, L/2, N)
thx, thy = np.meshgrid(th, th, indexing="xy")
extent = [th.min(), th.max(), th.min(), th.max()]

psi_f = "psi.npy"
BETA0_X, BETA0_Y = 0.15, -0.10

try:
    from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter
except Exception:
    gaussian_filter = maximum_filter = minimum_filter = None

def fermat_potential(psi, beta0x, beta0y):
    return 0.5*((thx - beta0x)**2 + (thy - beta0y)**2) - psi

def find_stationary_points(phiF, grad_thresh=None, nms_size=5):
    phiFx, phiFy = np.gradient(phiF, dth, edge_order=2)
    phiFxx, phiFxy = np.gradient(phiFx, dth, edge_order=2)
    phiFyx, phiFyy = np.gradient(phiFy, dth, edge_order=2)
    phiFxy = 0.5*(phiFxy + phiFyx)

    grad_mag = np.hypot(phiFx, phiFy)
    if grad_thresh is None:
        grad_thresh = np.quantile(grad_mag, 0.02)
    cand = grad_mag <= grad_thresh

    if gaussian_filter is not None:
        phiFs = gaussian_filter(phiF, 1.0)
    else:
        phiFs = phiF

    if maximum_filter is not None and minimum_filter is not None:
        loc_min = (phiFs == minimum_filter(phiFs, size=nms_size)) & cand
        loc_max = (phiFs == maximum_filter(phiFs, size=nms_size)) & cand
        base = cand & ~(loc_min | loc_max)
    else:
        loc_min = loc_max = np.zeros_like(cand, dtype=bool)
        base = np.zeros_like(cand, dtype=bool)
        base[::nms_size, ::nms_size] = cand[::nms_size, ::nms_size]

    detH = phiFxx*phiFyy - phiFxy**2
    trH  = phiFxx + phiFyy

    minima_mask = loc_min | (base & (detH > 0) & (trH > 0))
    maxima_mask = loc_max | (base & (detH > 0) & (trH < 0))
    saddle_mask = base & (detH < 0)

    mins_y, mins_x = np.nonzero(minima_mask)
    maxs_y, maxs_x = np.nonzero(maxima_mask)
    sads_y, sads_x = np.nonzero(saddle_mask)

    mins = [(float(thx[y,x]), float(thy[y,x])) for y,x in zip(mins_y, mins_x)]
    maxs = [(float(thx[y,x]), float(thy[y,x])) for y,x in zip(maxs_y, maxs_x)]
    sads = [(float(thx[y,x]), float(thy[y,x])) for y,x in zip(sads_y, sads_x)]
    return mins, sads, maxs

def main():
    if not os.path.exists(psi_f):
        raise FileNotFoundError("Missing psi.npy")
    psi = np.load(psi_f)

    phiF = fermat_potential(psi, BETA0_X, BETA0_Y)
    mins, sads, maxs = find_stationary_points(phiF, nms_size=7)

    # Compute gradient magnitude
    phiFx, phiFy = np.gradient(phiF, dth, edge_order=2)
    grad_mag = np.hypot(phiFx, phiFy)

    # ---------------------
    # Plot in subplots
    # ---------------------
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Fermat potential with contours + stationary points
    im0 = axs[0].imshow(phiF, origin="lower", extent=extent, cmap="magma")
    levels = np.linspace(np.nanmin(phiF), np.nanmax(phiF), 12)
    cs = axs[0].contour(thx, thy, phiF, levels=levels, colors="k", linewidths=0.6)
    axs[0].clabel(cs, inline=True, fontsize=7, fmt="")
    if mins: axs[0].plot(*zip(*mins), 'go', ms=6, label="minima")
    if sads: axs[0].plot(*zip(*sads), 'rx', ms=6, mew=1.5, label="saddles")
    if maxs: axs[0].plot(*zip(*maxs), 'bs', ms=5, mfc='none', label="maxima")
    axs[0].plot([BETA0_X], [BETA0_Y], 'w*', ms=10, mec='k', mew=0.8, label="beta0")
    axs[0].set_title("Fermat potential φF(θ)")
    axs[0].set_xlabel("θx [arcsec]"); axs[0].set_ylabel("θy [arcsec]")
    axs[0].legend(fontsize=8, loc="upper right")
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    # Gradient magnitude
    im1 = axs[1].imshow(grad_mag, origin="lower", extent=extent, cmap="viridis")
    axs[1].set_title("|∇φF|")
    axs[1].set_xlabel("θx [arcsec]"); axs[1].set_ylabel("θy [arcsec]")
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

    # ---------------------
    # Save outputs
    # ---------------------
    np.save("fermat_phi.npy", phiF)
    results = {
        "beta0": [BETA0_X, BETA0_Y],
        "n_minima": len(mins),
        "n_saddles": len(sads),
        "n_maxima": len(maxs),
        "minima_positions_arcsec": mins,
        "saddle_positions_arcsec": sads,
        "maxima_positions_arcsec": maxs
    }
    with open("task8_fermat_points.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

main()
