
import numpy as np

def normalize01(x, eps=1e-8):
    x = x.astype(np.float32)
    m, M = x.min(), x.max()
    if M - m < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - m) / (M - m + eps)

def visual_to_cortical(field_visual, cortex_h=256, cortex_w=384, a=0.1):
    """
    Simple log-polar transform to approximate V1 retinotopy.
    field_visual: HxW array in [0,1] visual space
    Returns cortical image (cortex_h x cortex_w)
    a controls eccentricity scaling.
    """
    H, W = field_visual.shape
    yy, xx = np.mgrid[0:cortex_h, 0:cortex_w].astype(np.float32)
    Xc = (xx + 0.5) / cortex_w * 2 - 1
    Yc = (yy + 0.5) / cortex_h * 2 - 1

    rho = np.exp(np.sqrt(Xc**2 + Yc**2) / a) - 1.0
    theta = np.arctan2(Yc, Xc + 1e-8)

    rho_n = rho / (rho.max() + 1e-8)
    xv = 0.5 + rho_n * np.cos(theta) * 0.5
    yv = 0.5 + rho_n * np.sin(theta) * 0.5

    xv = np.clip(xv, 0, 0.9999) * (W - 1)
    yv = np.clip(yv, 0, 0.9999) * (H - 1)

    x0 = np.floor(xv).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, W - 1)
    y0 = np.floor(yv).astype(np.int32)
    y1 = np.clip(y0 + 1, 0, H - 1)

    wa = (x1 - xv) * (y1 - yv)
    wb = (xv - x0) * (y1 - yv)
    wc = (x1 - xv) * (yv - y0)
    wd = (xv - x0) * (yv - y0)

    Ia = field_visual[y0, x0]
    Ib = field_visual[y0, x1]
    Ic = field_visual[y1, x0]
    Id = field_visual[y1, x1]

    out = wa * Ia + wb * Ib + wc * Ic + wd * Id
    return normalize01(out)
