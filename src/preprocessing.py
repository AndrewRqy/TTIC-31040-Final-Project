import numpy as np
import cv2


def histogram_equalization(img):
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycc[:, :, 0] = cv2.equalizeHist(ycc[:, :, 0])
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


def clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    c = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycc[:, :, 0] = c.apply(ycc[:, :, 0])
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


def gaussian_smoothing(img, sigma=1.0):
    k = max(int(np.ceil(sigma * 6)) | 1, 3)  # nearest odd >= 6*sigma
    return cv2.GaussianBlur(img, (k, k), sigma)


def bilateral_filter(img, d=9, sigma_color=75, sigma_space=75):
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


def non_local_means(img, h=10, patch_size=7, search_window=21):
    assert patch_size % 2 == 1 and search_window % 2 == 1

    return cv2.fastNlMeansDenoisingColored(
        img, None, h=h, hColor=h,
        templateWindowSize=patch_size, searchWindowSize=search_window,
    )


def intensity_normalization(img):
    f = img.astype(np.float32)
    out = np.empty_like(f)

    for c in range(img.shape[2]):
        ch = f[:, :, c]
        lo, hi = ch.min(), ch.max()
        out[:, :, c] = ch if hi - lo < 1e-6 else (ch - lo) / (hi - lo) * 255.0

    return np.clip(out, 0, 255).astype(np.uint8)


def no_preprocessing(img):
    return img


PREPROCESSING_FUNCTIONS = {
    'none':      no_preprocessing,
    'hist_eq':   histogram_equalization,
    'clahe':     clahe,
    'gaussian':  gaussian_smoothing,
    'bilateral': bilateral_filter,
    'nlm':       non_local_means,
    'normalize': intensity_normalization,
}

PREPROCESSING_LABELS = {
    'none':      'No Preprocessing',
    'hist_eq':   'Histogram Equalization',
    'clahe':     'CLAHE',
    'gaussian':  'Gaussian Smoothing',
    'bilateral': 'Bilateral Filter',
    'nlm':       'Non-Local Means',
    'normalize': 'Intensity Normalization',
}


def apply_preprocessing(img, method, **params):
    assert method in PREPROCESSING_FUNCTIONS, \
        f"Unknown method '{method}'. Choose from: {list(PREPROCESSING_FUNCTIONS)}"

    return PREPROCESSING_FUNCTIONS[method](img, **params)
