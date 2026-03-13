import numpy as np


def rotation_error(R_est, R_gt):
    if R_est is None:
        return np.nan

    cos_a = np.clip((np.trace(R_est.T @ R_gt) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


def translation_error(t_est, t_gt):
    if t_est is None:
        return np.nan

    te = np.asarray(t_est, np.float64).ravel()
    tg = np.asarray(t_gt,  np.float64).ravel()
    ne, ng = np.linalg.norm(te), np.linalg.norm(tg)

    if ne < 1e-8 or ng < 1e-8:
        return np.nan

    # abs: translation sign is ambiguous in 2D-2D recovery
    return float(np.degrees(np.arccos(np.clip(abs(np.dot(te/ne, tg/ng)), -1.0, 1.0))))


def epipolar_residual(F, p1, p2):
    if F is None or len(p1) == 0:
        return np.nan

    n = len(p1)
    ones = np.ones((n, 1), np.float64)
    q1 = np.hstack([p1.astype(np.float64), ones])
    q2 = np.hstack([p2.astype(np.float64), ones])

    l2 = (F   @ q1.T).T
    l1 = (F.T @ q2.T).T

    d2 = np.abs((q2 * l2).sum(1)) / np.sqrt(l2[:,0]**2 + l2[:,1]**2 + 1e-8)
    d1 = np.abs((q1 * l1).sum(1)) / np.sqrt(l1[:,0]**2 + l1[:,1]**2 + 1e-8)
    return float(np.mean(d2 + d1))


def fundamental_from_essential(E, K):
    Ki = np.linalg.inv(K)
    return Ki.T @ E @ Ki


def compute_metrics(res, R_gt, t_gt, p1=None, p2=None, K=None):
    epi = np.nan
    if p1 is not None and p2 is not None and res.get('success'):
        m = res.get('inlier_mask')
        if m is not None and m.sum() >= 4:
            q1, q2 = p1[m[:len(p1)]], p2[m[:len(p2)]]
            if res.get('E') is not None and K is not None:
                epi = epipolar_residual(fundamental_from_essential(res['E'], K), q1, q2)

    return {
        'rotation_error':    rotation_error(res.get('R'), R_gt),
        'translation_error': translation_error(res.get('t'), t_gt),
        'epipolar_residual': epi,
        'inlier_ratio':      res.get('inlier_ratio', np.nan),
        'num_matches':       res.get('num_matches', 0),
        'success':           res.get('success', False),
    }
