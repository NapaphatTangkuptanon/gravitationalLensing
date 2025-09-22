import os, json
import numpy as np
import matplotlib.pyplot as plt

L = 6.0
N = 512
dθ = L / N
θ = np.linspace(-L/2, L/2, N)
θx, θy = np.meshgrid(θ, θ, indexing="xy")
extent = [θ.min(), θ.max(), θ.min(), θ.max()]

try:
    from scipy.ndimage import gaussian_filter, label, center_of_mass, maximum_filter
except Exception:
    gaussian_filter = None
    label = None
    center_of_mass = None
    maximum_filter = None

def imshow_xy(P, title, cmap="viridis", vmin=None, vmax=None):
    plt.figure()
    plt.imshow(P, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    plt.xlabel("θx [arcsec]"); plt.ylabel("θy [arcsec]")
    plt.title(title); plt.colorbar()

def estimate_ring_radius(I, top_frac=0.01, smooth_sigma=1.0):
    if I is None:
        return float("nan"), float("nan")
    if gaussian_filter is not None and smooth_sigma > 0:
        Is = gaussian_filter(I, smooth_sigma)
    else:
        Is = I
    thr = np.quantile(Is, 1.0 - top_frac)
    mask = Is >= thr
    r = np.hypot(θx[mask], θy[mask])
    if r.size == 0:
        return float("nan"), float("nan")
    return float(np.median(r)), float(np.std(r))

def total_magnification(I_ray, S_unlensed):
    if I_ray is None or S_unlensed is None or S_unlensed.sum() <= 0:
        return float("nan")
    return float(I_ray.sum() / S_unlensed.sum())

def find_multiple_images(I, peak_frac=0.995, min_pixels=5):
    if I is None:
        return []
    thr = np.quantile(I, peak_frac)
    mask = I >= thr

    if label is not None and center_of_mass is not None:
        lab, n = label(mask)
        pts = []
        for k in range(1, n+1):
            mk = (lab == k)
            if mk.sum() < min_pixels:
                continue
            cy, cx = center_of_mass(mk)
            cy, cx = int(round(cy)), int(round(cx))
            pts.append((float(θx[cy, cx]), float(θy[cy, cx])))
        return pts

    if maximum_filter is not None:
        neighborhood = maximum_filter(I, size=5)
        peaks = (I == neighborhood) & mask
        ys, xs = np.nonzero(peaks)
        return [(float(θx[y, x]), float(θy[y, x])) for y, x in zip(ys, xs)]

    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return []
    cy, cx = int(np.median(ys)), int(np.median(xs))
    return [(float(θx[cy, cx]), float(θy[cy, cx]))]

def main():
    I_psf_f   = "I_psf.npy"
    I_noisy_f = "I_noisy.npy"
    I_ray_f   = "I_ray.npy"
    beta_x_f  = "beta_x.npy"
    beta_y_f  = "beta_y.npy"

    I_psf   = np.load(I_psf_f)   if os.path.exists(I_psf_f) else None
    I_noisy = np.load(I_noisy_f) if os.path.exists(I_noisy_f) else None
    I_ray   = np.load(I_ray_f)   if os.path.exists(I_ray_f) else None
    beta_x  = np.load(beta_x_f)  if os.path.exists(beta_x_f) else None
    beta_y  = np.load(beta_y_f)  if os.path.exists(beta_y_f) else None

    if I_psf is not None:
        image_for_radius = I_psf
    elif I_noisy is not None:
        image_for_radius = I_noisy
    elif I_ray is not None:
        image_for_radius = I_ray
    else:
        raise FileNotFoundError("Need at least one of I_psf.npy, I_noisy.npy, or I_ray.npy for Task 6.")

    ring_r, ring_std = estimate_ring_radius(image_for_radius, top_frac=0.01, smooth_sigma=1.0)
    M = float("nan")
    img_positions = find_multiple_images(I_ray if I_ray is not None else image_for_radius,
                                         peak_frac=0.995, min_pixels=5)

    print(f"Estimated Einstein ring radius : {ring_r:.3f} ± {ring_std:.3f} arcsec")
    print(f"Total magnification (no S_unlensed) : {M}")
    print("Multiple image positions (arcsec):")
    if len(img_positions) == 0:
        print("  none detected")
    else:
        for (x0, y0) in img_positions:
            print(f"  ({x0:+.3f}, {y0:+.3f})")

    out = dict(
        ring_radius_arcsec = ring_r,
        ring_radius_std    = ring_std,
        magnification      = M,
        n_images           = len(img_positions),
        image_positions    = img_positions
    )
    with open("task6.json", "w") as f:
        json.dump(out, f, indent=2)
    print("task6.json")

    plt.figure()
    plt.imshow(image_for_radius, origin="lower", extent=extent, cmap="magma")
    for (x0, y0) in img_positions:
        plt.plot([x0], [y0], 'wo', mfc='none', ms=6, mew=1.5)
    plt.xlabel("θx [arcsec]"); plt.ylabel("θy [arcsec]")
    plt.title("Multiple image positions")
    plt.colorbar()
    plt.show()

main()