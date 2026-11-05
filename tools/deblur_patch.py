"""Richardson-Lucy deconvolution of a patch of an equirect photo (luminance only), to win back detail that the lens-seam blend smeared.
Used by render_shot_032.py; no scipy needed (FFT with numpy)."""
import numpy as np

def gauss_psf(sigma, size=None):
    size = size or int(sigma * 6) | 1
    ax = np.arange(size) - size // 2
    k = np.exp(-(ax[:, None] ** 2 + ax[None, :] ** 2) / (2 * sigma * sigma)); return k / k.sum()

def _fft_conv(img, psf_f, shape):
    return np.real(np.fft.ifft2(np.fft.fft2(img) * psf_f))

def richardson_lucy(y, sigma=1.8, iters=14, damp=0.12):
    """y: 2D float array 0..255. Returns deconvolved array. Edges are mirrored to avoid wrap-around ringing."""
    pad = int(sigma * 6)
    yp = np.pad(y, pad, mode="reflect"); h, w = yp.shape
    psf = np.zeros((h, w)); k = gauss_psf(sigma); kh = k.shape[0]
    psf[:kh, :kh] = k; psf = np.roll(np.roll(psf, -(kh // 2), 0), -(kh // 2), 1)
    H = np.fft.fft2(psf); Hc = np.conj(H)
    obs = np.maximum(yp / 255.0, 1e-4); est = obs.copy()
    for _ in range(iters):
        conv = np.real(np.fft.ifft2(np.fft.fft2(est) * H))
        ratio = obs / np.maximum(conv, 1e-4)
        corr = np.real(np.fft.ifft2(np.fft.fft2(ratio) * Hc))
        upd = est * corr
        est = est + (1 - damp) * (upd - est)               # damped step: less ringing
    out = np.clip(est * 255.0, 0, 255)
    return out[pad:-pad, pad:-pad]

def deblur_rgb(rgb, sigma=1.8, iters=14, damp=0.12, strength=1.0):
    """rgb uint8 (h,w,3) -> uint8. Works on luminance; colour is kept."""
    f = rgb.astype(np.float32)
    Y = f @ np.array([0.299, 0.587, 0.114], np.float32)
    Y2 = richardson_lucy(Y.astype(np.float64), sigma, iters, damp).astype(np.float32)
    d = (Y2 - Y) * strength
    return np.clip(f + d[..., None], 0, 255).astype(np.uint8)
