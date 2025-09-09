import sys, math, json, typing, pathlib
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift


from skimage import transform as sktf, measure as skmeas, filters as skfilt
from tqdm import tqdm

import os

L = 6.0
psi_path = "psi.npy"

if os.path.exists(psi_path):
    psi = np.load(psi_path)
    N = psi.shape[0]
    dth = L / N
    print(f"Loaded ψ from {psi_path} (N={N})")
else:
    N   = 512
    dth = L / N
    thE = 1.2            # Einstein radius [arcsec]
    gamma_ext = 0.08     # external shear amplitude
    phi_deg   = 30.0     # external shear angle
    phi = np.deg2rad(phi_deg)
    g1  = gamma_ext*np.cos(2*phi)
    g2  = gamma_ext*np.sin(2*phi)

    x = (np.arange(N) - N//2) * dth
    X, Y = np.meshgrid(x, x, indexing='xy')
    r = np.hypot(X, Y) + 1e-12
    psi = thE*r + 0.5*(g1*(X**2 - Y**2) + 2*g2*X*Y)
    print("Built analytic ψ (SIS + external shear) as a fallback.")

x = (np.arange(N) - N//2) * dth
X, Y = np.meshgrid(x, x, indexing='xy')
extent = [-L/2, L/2, -L/2, L/2]

# Deflection → α = ∇ψ
psi_x, psi_y = np.gradient(psi, dth, edge_order=2)
alpha_x, alpha_y = psi_x, psi_y
alpha_mag = np.hypot(alpha_x, alpha_y)

# Second derivatives → κ, γ
psi_xx, psi_xy = np.gradient(psi_x, dth, edge_order=2)
psi_yx, psi_yy = np.gradient(psi_y, dth, edge_order=2)

kappa   = 0.5*(psi_xx + psi_yy)
gamma1  = 0.5*(psi_xx - psi_yy)
gamma2  = psi_xy
gamma_mag = np.hypot(gamma1, gamma2)

# Magnification and critical curve (***det A = 0***)
detA = (1 - kappa)**2 - gamma_mag**2
with np.errstate(divide='ignore', invalid='ignore'):
    mu = 1.0 / detA

#Deflection vec.
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# --- Deflection magnitude ---
im0 = axs[0,0].imshow(alpha_mag, origin='lower', extent=extent)
axs[0,0].set_title("Deflection |α|")
axs[0,0].set_xlabel("θx [arcsec]"); axs[0,0].set_ylabel("θy [arcsec]")
step = max(1, N//32)
axs[0,0].quiver(X[::step,::step], Y[::step,::step],
                alpha_x[::step,::step], alpha_y[::step,::step],
                angles='xy', scale_units='xy', scale=1.0, width=0.002)
fig.colorbar(im0, ax=axs[0,0], fraction=0.046, pad=0.04)

# --- Shear magnitude ---
im1 = axs[0,1].imshow(gamma_mag, origin='lower', extent=extent)
axs[0,1].set_title("Shear |γ|")
axs[0,1].set_xlabel("θx [arcsec]"); axs[0,1].set_ylabel("θy [arcsec]")
phi_g = 0.5*np.arctan2(gamma2[::step,::step], gamma1[::step,::step])
gm = np.hypot(gamma1[::step,::step], gamma2[::step,::step])
axs[0,1].quiver(X[::step,::step], Y[::step,::step],
                gm*np.cos(phi_g), gm*np.sin(phi_g),
                angles='xy', scale_units='xy', scale=1.0, width=0.002)
fig.colorbar(im1, ax=axs[0,1], fraction=0.046, pad=0.04)

# --- Convergence κ with critical curve ---
im2 = axs[1,0].imshow(kappa, origin='lower', extent=extent)
axs[1,0].set_title("Convergence κ with critical curve")
axs[1,0].set_xlabel("θx [arcsec]"); axs[1,0].set_ylabel("θy [arcsec]")
axs[1,0].contour(X, Y, detA, levels=[0.0], colors='k', linewidths=1.2)
fig.colorbar(im2, ax=axs[1,0], fraction=0.046, pad=0.04)

# --- Magnification μ ---
im3 = axs[1,1].imshow(np.clip(mu, -50, 50), origin='lower', extent=extent)
axs[1,1].set_title("Magnification μ (clipped)")
axs[1,1].set_xlabel("θx [arcsec]"); axs[1,1].set_ylabel("θy [arcsec]")
fig.colorbar(im3, ax=axs[1,1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()
SAVE = True
if SAVE:
    np.save("alpha_x.npy", alpha_x);  np.save("alpha_y.npy", alpha_y);  np.save("alpha_mag.npy", alpha_mag)
    np.save("kappa_from_psi.npy", kappa)
    np.save("gamma1.npy", gamma1);    np.save("gamma2.npy", gamma2);    np.save("gamma_mag.npy", gamma_mag)
    np.save("mu_from_psi.npy", mu);   np.save("detA_from_psi.npy", detA)