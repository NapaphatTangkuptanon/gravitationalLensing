import numpy as np
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt

def gaussian_psf(N, dtheta, fwhm):
    sigma = fwhm / (2*np.sqrt(2*np.log(2)))
    x = (np.arange(N) - N//2) * dtheta
    X, Y = np.meshgrid(x, x, indexing="xy")
    psf = np.exp(-(X**2 + Y**2)/(2*sigma**2))
    return psf / psf.sum()

def apply_instrument(I_lensed, dtheta, fwhm=0.12, sigma_read=3.0, seed=42):
    N = I_lensed.shape[0]
    psf = gaussian_psf(N, dtheta, fwhm)
    Iconv = fftconvolve(I_lensed, psf, mode="same")

    rng = np.random.default_rng(seed)
    I_counts = rng.poisson(np.clip(Iconv, 0, None))
    I_noisy = I_counts + rng.normal(0, sigma_read, I_counts.shape)

    return Iconv, I_noisy

if __name__ == "__main__":
    N, L = 256, 6.0
    dth = L/N
    x = (np.arange(N)-N//2)*dth
    X,Y = np.meshgrid(x,x,indexing="xy")
    I_test = np.exp(-(X**2+Y**2)/(2*0.1**2))*5000

    Iconv, I_noisy = apply_instrument(I_test, dth)

    fig, axs = plt.subplots(1,3,figsize=(12,4))
    axs[0].imshow(I_test, origin="lower"); axs[0].set_title("Ideal lensed image")
    axs[1].imshow(Iconv, origin="lower"); axs[1].set_title("After PSF convolution")
    axs[2].imshow(I_noisy, origin="lower"); axs[2].set_title("Final noisy image")
    for ax in axs: ax.axis("off")
    plt.tight_layout(); plt.show()
