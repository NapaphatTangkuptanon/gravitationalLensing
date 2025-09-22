import os
import numpy as np
import matplotlib.pyplot as plt

# -------------------
# Setup / grids
# -------------------
L = 6.0
N = 512
dθ = L / N
θ = np.linspace(-L/2, L/2, N)
θx, θy = np.meshgrid(θ, θ, indexing="xy")
extent = [θ.min(), θ.max(), θ.min(), θ.max()]

# -------------------
# File names
# -------------------
kappa_f   = "kappa.npy"
psi_f     = "psi.npy"
alpha_x_f = "alpha_x.npy"
alpha_y_f = "alpha_y.npy"
mu_f      = "mu_from_psi.npy"
beta_x_f  = "beta_x.npy"
beta_y_f  = "beta_y.npy"
I_noisy_f = "I_noisy.npy"
I_psf_f   = "I_psf.npy"
I_ray_f   = "I_ray.npy"

# -------------------
# Helper: imshow on axis with colorbar
# -------------------
def imshow_ax(ax, P, title, cmap="viridis", vmin=None, vmax=None):
    im = ax.imshow(P, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_title(title)
    ax.set_xlabel("θx [arcsec]"); ax.set_ylabel("θy [arcsec]")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return im, cbar

def main():
    # -------------------
    # Load κ and ψ
    # -------------------
    if not os.path.exists(kappa_f) or not os.path.exists(psi_f):
        raise FileNotFoundError("Need kappa.npy and psi.npy from Task 1.")

    kappa = np.load(kappa_f)
    psi   = np.load(psi_f)

    # -------------------
    # Deflection α
    # -------------------
    if os.path.exists(alpha_x_f) and os.path.exists(alpha_y_f):
        alpha_x = np.load(alpha_x_f)
        alpha_y = np.load(alpha_y_f)
    else:
        psi_x, psi_y = np.gradient(psi, dθ, edge_order=2)
        alpha_x, alpha_y = psi_x, psi_y  # α = ∇ψ

    alpha_mag = np.hypot(alpha_x, alpha_y)

    # -------------------
    # Magnification μ (load or compute)
    # -------------------
    if os.path.exists(mu_f):
        mu = np.load(mu_f)
    else:
        psi_x, psi_y = np.gradient(psi, dθ, edge_order=2)
        psi_xx, psi_xy = np.gradient(psi_x, dθ, edge_order=2)
        psi_yx, psi_yy = np.gradient(psi_y, dθ, edge_order=2)
        psi_xy = 0.5*(psi_xy + psi_yx)  # symmetrize
        kappa_from_psi = 0.5*(psi_xx + psi_yy)
        gamma1 = 0.5*(psi_xx - psi_yy)
        gamma2 = psi_xy
        detA = (1.0 - kappa_from_psi)**2 - (gamma1**2 + gamma2**2)
        with np.errstate(divide="ignore", invalid="ignore"):
            mu = 1.0 / detA
            mu[~np.isfinite(mu)] = 0.0

    # -------------------
    # β mapping (save for reuse)
    # -------------------
    if os.path.exists(beta_x_f) and os.path.exists(beta_y_f):
        beta_x = np.load(beta_x_f)
        beta_y = np.load(beta_y_f)
    else:
        # θ-grid in arcsec
        x = (np.arange(N) - N//2) * dθ
        X, Y = np.meshgrid(x, x, indexing="xy")
        beta_x = X - alpha_x
        beta_y = Y - alpha_y
        np.save(beta_x_f, beta_x)
        np.save(beta_y_f, beta_y)
        print("[Task 3] Saved beta_x.npy and beta_y.npy")
    beta_mag = np.hypot(beta_x, beta_y)

    # -------------------
    # Final image (choose best available)
    # -------------------
    if os.path.exists(I_noisy_f):
        I = np.load(I_noisy_f); I_title = "Final Lensed Image (noisy)"
    elif os.path.exists(I_psf_f):
        I = np.load(I_psf_f);    I_title = "PSF-convolved Lensed Image"
    elif os.path.exists(I_ray_f):
        I = np.load(I_ray_f);    I_title = "Ray-traced Lensed Image"
    else:
        # Minimal fallback: Gaussian source in β-plane, sampled back on θ-plane
        I = np.exp(-0.5*((beta_x/0.2)**2 + (beta_y/0.2)**2))
        np.save(I_ray_f, I)
        I_title = "Ray-traced Lensed Image (fallback)"
        print("[Task 3] Saved I_ray.npy (fallback).")

    # -------------------
    # Plot grid (2×3)
    # -------------------
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    # κ
    imshow_ax(axs[0,0], kappa, "Convergence κ(θ)")

    # ψ with contours
    im, _ = imshow_ax(axs[0,1], psi, "Potential ψ(θ) with contours", cmap="Greys")
    cs = axs[0,1].contour(θx, θy, psi, colors="C1", linewidths=0.8)
    axs[0,1].clabel(cs, inline=True, fontsize=8)

    # |α|
    imshow_ax(axs[0,2], alpha_mag, "Deflection |α|")
    step = max(1, N//32)
    axs[0,2].quiver(θx[::step,::step], θy[::step,::step],
                    alpha_x[::step,::step], alpha_y[::step,::step],
                    angles='xy', scale_units='xy', scale=1.0, width=0.002)

    # μ (clipped)
    imshow_ax(axs[1,0], np.clip(mu, -10, 10), "Magnification μ (clipped)", cmap="coolwarm")

    # |β|
    imshow_ax(axs[1,1], beta_mag, "Mapped Source-plane |β|", cmap="viridis")

    # Final image
    imshow_ax(axs[1,2], I, I_title, cmap="magma")

    plt.tight_layout()
    plt.show()

main()
