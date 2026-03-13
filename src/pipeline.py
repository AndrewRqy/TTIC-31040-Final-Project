import numpy as np
import cv2
from .preprocessing import apply_preprocessing


def detect_and_match_sift(img1, img2, ratio=0.75, n_feat=3000):
    sift = cv2.SIFT_create(nfeatures=n_feat)
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if img1.ndim == 3 else img1
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if img2.ndim == 3 else img2

    kp1, d1 = sift.detectAndCompute(g1, None)
    kp2, d2 = sift.detectAndCompute(g2, None)

    _empty = ([], [], np.zeros((0,2), np.float32), np.zeros((0,2), np.float32),
              np.zeros((0,128), np.float32), np.zeros((0,128), np.float32))
    if d1 is None or d2 is None or len(kp1) < 8 or len(kp2) < 8:
        return _empty

    raw = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2)
    p1, p2, e1, e2 = [], [], [], []
    for m, n in raw:
        if m.distance < ratio * n.distance:
            p1.append(kp1[m.queryIdx].pt)
            p2.append(kp2[m.trainIdx].pt)
            e1.append(d1[m.queryIdx])
            e2.append(d2[m.trainIdx])

    if len(p1) < 8:
        return kp1, kp2, np.zeros((0,2), np.float32), np.zeros((0,2), np.float32), \
               np.zeros((0,128), np.float32), np.zeros((0,128), np.float32)

    return kp1, kp2, np.array(p1, np.float32), np.array(p2, np.float32), \
           np.array(e1, np.float32), np.array(e2, np.float32)


def _pose_from_E(E, p1, p2, K):
    _, R, t, mask = cv2.recoverPose(E, p1, p2, K)
    mask = np.ones(len(p1), dtype=bool) if mask is None else mask.ravel().astype(bool)
    return R, t.ravel(), mask


def _pose_from_H(H, K):
    _, Rs, ts, _ = cv2.decomposeHomographyMat(H, K)
    t = ts[0].ravel()
    return Rs[0], t / (np.linalg.norm(t) + 1e-8)


def run_essential_pipeline(img1, img2, K, ratio=0.75, thresh=1.0, conf=0.999, n_feat=3000):
    res = dict(num_matches=0, inlier_ratio=0.0, R=None, t=None, E=None, success=False)

    _, _, p1, p2, _, _ = detect_and_match_sift(img1, img2, ratio, n_feat)
    res['num_matches'] = len(p1)

    if len(p1) < 8:
        res['inlier_mask'] = np.zeros(0, dtype=bool)
        return res

    E, mask = cv2.findEssentialMat(p1, p2, K, method=cv2.RANSAC, prob=conf, threshold=thresh)
    if E is None or E.shape != (3, 3):
        res['inlier_mask'] = np.zeros(len(p1), dtype=bool)
        return res

    m = mask.ravel().astype(bool)
    res['inlier_ratio'] = m.sum() / len(p1)
    res['E'] = E
    p1_in, p2_in = p1[m], p2[m]

    if len(p1_in) < 5:
        res['inlier_mask'] = m
        return res

    R, t, pmask = _pose_from_E(E, p1_in, p2_in, K)
    res['R'], res['t'] = R, t

    # propagate cheirality mask back to full index
    full = np.zeros(len(p1), dtype=bool)
    full[m] = pmask
    res['inlier_mask'] = full
    res['success'] = True
    return res


def run_homography_pipeline(img1, img2, K, ratio=0.75, thresh=3.0, conf=0.995, n_feat=3000):
    res = dict(num_matches=0, inlier_ratio=0.0, R=None, t=None, H=None, success=False)

    _, _, p1, p2, _, _ = detect_and_match_sift(img1, img2, ratio, n_feat)
    res['num_matches'] = len(p1)

    if len(p1) < 4:
        res['inlier_mask'] = np.zeros(0, dtype=bool)
        return res

    H, mask = cv2.findHomography(p1, p2, cv2.RANSAC, thresh, confidence=conf)
    if H is None:
        res['inlier_mask'] = np.zeros(len(p1), dtype=bool)
        return res

    m = mask.ravel().astype(bool)
    res['inlier_ratio'] = m.sum() / len(p1)
    res['H'] = H
    res['inlier_mask'] = m

    res['R'], res['t'] = _pose_from_H(H, K)
    res['success'] = True
    return res


def run_pipeline(img1, img2, K, model='essential',
                 preprocess_method=None, preprocess_params=None,
                 ratio=0.75, thresh=1.0, conf=0.999, n_feat=3000):
    assert model in ('essential', 'homography')

    if preprocess_method and preprocess_method != 'none':
        kw = preprocess_params or {}
        img1 = apply_preprocessing(img1, preprocess_method, **kw)
        img2 = apply_preprocessing(img2, preprocess_method, **kw)

    fn = run_essential_pipeline if model == 'essential' else run_homography_pipeline
    return fn(img1, img2, K, ratio=ratio, thresh=thresh, conf=conf, n_feat=n_feat)
