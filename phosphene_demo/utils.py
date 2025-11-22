
import numpy as np
import cv2

def normalize01(x, eps=1e-8):
    x = x.astype(np.float32)
    m, M = x.min(), x.max()
    if M - m < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - m) / (M - m + eps)

def make_gaussian_field(points, weights, H, W, sigma=0.04):
    """
    points: (K, 2) in normalized visual coordinates [0,1]x[0,1], (x,y)
    weights: (K,) intensity 0..1
    sigma: spread in normalized units (relative to field size)
    """
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    X = xx / (W - 1 + 1e-8)
    Y = yy / (H - 1 + 1e-8)
    field = np.zeros((H, W), dtype=np.float32)
    s2 = (sigma**2)
    for (x, y), w in zip(points, weights):
        d2 = (X - x)**2 + (Y - y)**2
        field += w * np.exp(-0.5 * d2 / s2)
    return normalize01(field)

def grid_points(n):
    xs = (np.arange(n) + 0.5) / n
    ys = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(xs, ys)
    pts = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1)
    return pts  # (n*n, 2)

def draw_tiled(imgs, scale=1.0, labels=None):
    # pick reference height (first image)
    ref_h = imgs[0].shape[0]
    out = []
    for i, im in enumerate(imgs):
        if im.ndim == 2:
            im = cv2.cvtColor((normalize01(im)*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        else:
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        # resize to match first image height
        if im.shape[0] != ref_h:
            im = cv2.resize(im, (int(im.shape[1]*ref_h/im.shape[0]), ref_h),
                            interpolation=cv2.INTER_AREA)
        if labels is not None:
            cv2.putText(im, labels[i], (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        out.append(im)
    row = np.concatenate(out, axis=1)
    if scale != 1.0:
        row = cv2.resize(row, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return row

def downsample_gray(frame_rgb, n, gamma=0.8):
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    small = cv2.resize(gray, (n, n), interpolation=cv2.INTER_AREA)
    small = np.power(small, gamma)
    return small  # (n,n) 0..1
