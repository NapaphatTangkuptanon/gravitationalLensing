import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve


L = 6.0         
N = 512         
dth = L / N
theta_E = 1.2   
gamma_ext = 0.08
phi_gamma_deg = 30.0

psi_path = "psi.npy"
if os.path.exists(psi_path):
    psi = np.load(psi_path)
    print(f"Loaded ψ from {psi_path}")
else:
    x = (np.arange(N)-N//2)*dth
    X,Y = np.meshgrid(x,x,indexing="xy")
    r = np.hypot(X,Y)+1e-12
    phi_g = np.deg2rad(phi_gamma_deg)
    g1 = gamma_ext*np.cos(2*phi_g)
    g2 = gamma_ext*np.sin(2*phi_g)
    psi = theta_E*r + 0.5*(g1*(X**2-Y**2)+2*g2*X*Y)
    print("Built analytic ψ fallback (SIS + shear).")

psi_x, psi_y = np.gradient(psi, dth, edge_order=2)
alpha_x, alpha_y = psi_x, psi_y

def sersic_source(beta_x, beta_y, n=1.0, Re=0.25, q=0.7, phi_deg=-20.0,
                  beta0=(0.15,-0.10), I0=5000.0):
    phi = np.deg2rad(phi_deg)
    c,s = np.cos(-phi), np.sin(-phi)
    Xr = c*(beta_x-beta0[0]) - s*(beta_y-beta0[1])
    Yr = s*(beta_x-beta0[0]) + c*(beta_y-beta0[1])
    r_ell = np.sqrt(Xr**2 + (Yr/q)**2)
    b_n = 2*n - 1/3 + 4/(405*n) + 46/(25515*n**2)
    return I0 * np.exp(-b_n*((r_ell/Re)**(1/n)-1))

Ns = 512
Lsrc = 3.0
ds = Lsrc/Ns
xs = (np.arange(Ns)-Ns//2)*ds
Xb,Yb = np.meshgrid(xs,xs,indexing="xy")
Strue = sersic_source(Xb,Yb)

x = (np.arange(N)-N//2)*dth
X,Y = np.meshgrid(x,x,indexing="xy")
beta_x = X - alpha_x
beta_y = Y - alpha_y

# Map to source grid indices
ix = np.clip(((beta_x+Lsrc/2)/ds).astype(int),0,Ns-1)
iy = np.clip(((beta_y+Lsrc/2)/ds).astype(int),0,Ns-1)
I_lensed = Strue[iy,ix]

def gaussian_psf(N, d, fwhm):
    sigma = fwhm/(2*np.sqrt(2*np.log(2)))
    x = (np.arange(N)-N//2)*d
    X,Y = np.meshgrid(x,x,indexing="xy")
    psf = np.exp(-(X**2+Y**2)/(2*sigma**2))
    return psf/psf.sum()

psf = gaussian_psf(N, dth, 0.12)
Iconv = fftconvolve(I_lensed, psf, mode="same")

rng = np.random.default_rng(42)
I_counts = rng.poisson(np.clip(Iconv,0,None))
I_noisy = I_counts + rng.normal(0,3.0,I_counts.shape)

plt.figure(figsize=(5,5))
plt.imshow(Strue, origin="lower", extent=[-Lsrc/2,Lsrc/2,-Lsrc/2,Lsrc/2])
plt.xlabel("βx [arcsec]"); plt.ylabel("βy [arcsec]")
plt.title("Unlensed Sérsic source")
plt.colorbar(label="counts")
plt.tight_layout(); plt.show()

plt.figure(figsize=(6,6))
plt.imshow(I_noisy, origin="lower", extent=[-L/2,L/2,-L/2,L/2])
plt.xlabel("θx [arcsec]"); plt.ylabel("θy [arcsec]")
plt.title("Noisy lensed image")
plt.colorbar(label="counts")
plt.tight_layout(); plt.show()
