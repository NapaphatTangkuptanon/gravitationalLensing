import sys, math, json, typing, pathlib
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift


from skimage import transform as sktf, measure as skmeas, filters as skfilt
from tqdm import tqdm


#Guass_Seidel

L = 6.0
N = 512
dθ = L / N
θE = 1.2 #Einstein radius

x = (np.arange(N) - N//2) * dθ
X, Y = np.meshgrid(x, x, indexing='xy')
r = np.sqrt(X*X + Y*Y) + 1e-12
kappa = θE / (2.0 * r)

def sor_poisson_rb(kappa, d, omega, tol, max_iter=10000):
  n = kappa.shape[0]
  psi = np.zeros_like(kappa)

  rhs = 2.0 * kappa * (d*d)

  i0, i1 = 1, n-1
  j0, j1 = 1, n-1
  for it in range(1, max_iter+1):
      for i in range(i0, i1):
          for j in range(j0, j1):
            if((i+j)&1==0):
              S = psi[i+1,j] + psi[i-1,j] + psi[i,j+1] + psi[i,j-1]
              psi[i,j] = (1.0 - omega)*psi[i,j] + 0.25*omega*(S - rhs[i,j])

      for i in range(i0, i1):
          for j in range(j0, j1):
            if((i+j)&1==1):
              S = psi[i+1,j] + psi[i-1,j] + psi[i,j+1] + psi[i,j-1]
              psi[i,j] = (1.0 - omega)*psi[i,j] + 0.25*omega*(S - rhs[i,j])
      max_res = 0.0
      invd2 = 1.0 / (d*d)
      for i in range(i0, i1):
          for j in range(j0, j1):
            lap = (psi[i+1,j] + psi[i-1,j] + psi[i,j+1] + psi[i,j-1] - 4.0*psi[i,j]) * invd2
            res = lap - 2.0*kappa[i,j]
            ares = abs(res)
            if ares > max_res:
                max_res = ares

      if max_res < tol:
          return psi, it, max_res
  return psi, max_iter, max_res

omega = 1.99
tol = 1e-6

psi, iter, final_res = sor_poisson_rb(kappa, dθ, omega=omega, tol=tol)

print(f"SOR converged in {iter} iterations, final max|res| = {final_res:.3e}  (ω={omega}, N={N})")

plt.figure(figsize=(6,6))
plt.contourf(X, Y, psi, levels=20)
plt.xlabel("θx [arcsec]"); plt.ylabel("θy [arcsec]")
plt.title("Lensing potential ψ (contours)")
plt.colorbar()
plt.tight_layout()
plt.show()

np.save("psi.npy",   psi)
np.save("kappa.npy", kappa)