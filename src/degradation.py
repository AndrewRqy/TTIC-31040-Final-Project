import numpy as np
import cv2

_SIGMAS  = [10, 25, 50, 80, 120]
_KERNELS = [7, 15, 25, 35, 51]
_JPEGS   = [75, 50, 30, 15, 5]
_GAMMAS  = [0.55, 0.35, 0.20, 0.10, 0.05]


def apply_gaussian_noise(img, severity=1):
    assert 1 <= severity <= 5

    noise = np.random.randn(*img.shape).astype(np.float32) * _SIGMAS[severity - 1]
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def apply_motion_blur(img, severity=1, angle=0.0):
    assert 1 <= severity <= 5

    k = _KERNELS[severity - 1]
    ker = np.zeros((k, k), dtype=np.float32)
    ker[k // 2, :] = 1.0 / k

    if angle != 0.0:
        M = cv2.getRotationMatrix2D((k // 2, k // 2), angle, 1.0)
        ker = cv2.warpAffine(ker, M, (k, k))
        s = ker.sum()
        if s > 1e-8:
            ker /= s

    return cv2.filter2D(img, -1, ker)


def apply_jpeg_compression(img, severity=1):
    assert 1 <= severity <= 5

    _, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), _JPEGS[severity - 1]])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def apply_illumination_shift(img, severity=1):
    assert 1 <= severity <= 5

    g = _GAMMAS[severity - 1]
    lut = np.array([((i / 255.0) ** g) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, lut)


DEGRADATION_FUNCTIONS = {
    'gaussian_noise': apply_gaussian_noise,
    'motion_blur':    apply_motion_blur,
    'jpeg':           apply_jpeg_compression,
    'illumination':   apply_illumination_shift,
}

DEGRADATION_LABELS = {
    'gaussian_noise': 'Gaussian Noise',
    'motion_blur':    'Motion Blur',
    'jpeg':           'JPEG Compression',
    'illumination':   'Illumination Shift',
}


def apply_degradation(img, deg_type, severity=1):
    assert deg_type in DEGRADATION_FUNCTIONS, \
        f"Unknown degradation '{deg_type}'. Choose from: {list(DEGRADATION_FUNCTIONS)}"
    assert 1 <= severity <= 5

    return DEGRADATION_FUNCTIONS[deg_type](img, severity=severity)


def get_severity_label(deg_type, severity):
    return {
        'gaussian_noise': f"σ={_SIGMAS[severity-1]}",
        'motion_blur':    f"k={_KERNELS[severity-1]}",
        'jpeg':           f"q={_JPEGS[severity-1]}",
        'illumination':   f"γ={_GAMMAS[severity-1]}",
    }[deg_type]
