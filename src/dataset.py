# TUM RGB-D groundtruth format: timestamp tx ty tz qx qy qz qw
# fr3 is pre-rectified (zero distortion); fr1/fr2 need undistortion.

import numpy as np
import cv2
from pathlib import Path
from scipy.spatial.transform import Rotation


SEQUENCE_PARAMS = {
    'fr1': {'fx': 517.3, 'fy': 516.5, 'cx': 318.6, 'cy': 255.3,
            'dist': np.array([0.2624, -0.9531, -0.0054, 0.0026, 1.1633])},
    'fr2': {'fx': 520.9, 'fy': 521.0, 'cx': 325.1, 'cy': 249.7,
            'dist': np.array([0.2312, -0.7849, -0.0033, -0.0001, 0.9172])},
    'fr3': {'fx': 535.4, 'fy': 539.2, 'cx': 320.1, 'cy': 247.6,
            'dist': np.zeros(5)},
}


def _quat_to_rot(qx, qy, qz, qw):
    return Rotation.from_quat([qx, qy, qz, qw]).as_matrix()


def _pose_to_T(tx, ty, tz, qx, qy, qz, qw):
    T = np.eye(4)
    T[:3, :3] = _quat_to_rot(qx, qy, qz, qw)
    T[:3, 3]  = [tx, ty, tz]
    return T


def _associate(ts_q, ts_r, max_diff=0.02):
    ts_q, ts_r = np.asarray(ts_q), np.asarray(ts_r)
    idx = np.searchsorted(ts_r, ts_q)
    matched = np.full(len(ts_q), -1, dtype=int)

    for i, (t, j) in enumerate(zip(ts_q, idx)):
        cands = [c for c in (j-1, j) if 0 <= c < len(ts_r)]
        if not cands:
            continue
        best = min(cands, key=lambda c: abs(ts_r[c] - t))
        if abs(ts_r[best] - t) <= max_diff:
            matched[i] = best

    return matched


class TUMDataset:
    def __init__(self, root_dir, seq_type='fr1', stride=5, undistort=True):
        assert seq_type in SEQUENCE_PARAMS
        self.root      = Path(root_dir)
        self.params    = SEQUENCE_PARAMS[seq_type]
        self.stride    = stride
        self.undistort = undistort
        self.K = self._build_K()
        self._load()

    def _build_K(self):
        p = self.params
        return np.array([[p['fx'], 0., p['cx']],
                         [0., p['fy'], p['cy']],
                         [0.,     0.,     1.]], dtype=np.float64)

    def _load(self):
        rgb_path = self.root / 'rgb.txt'
        gt_path  = self.root / 'groundtruth.txt'
        assert rgb_path.exists() and gt_path.exists()

        rgb_ts, rgb_files = [], []
        for ln in rgb_path.read_text().splitlines():
            if ln.startswith('#') or not ln.strip(): continue
            p = ln.split()
            rgb_ts.append(float(p[0]))
            rgb_files.append(p[1])

        gt_ts, gt_poses = [], []
        for ln in gt_path.read_text().splitlines():
            if ln.startswith('#') or not ln.strip(): continue
            p = list(map(float, ln.split()))
            gt_ts.append(p[0])
            gt_poses.append(p[1:])

        rgb_ts   = np.array(rgb_ts)
        gt_ts    = np.array(gt_ts)
        gt_poses = np.array(gt_poses)

        matched = _associate(rgb_ts, gt_ts)
        valid   = matched >= 0
        self._files = [rgb_files[i] for i in np.where(valid)[0]]
        self._Ts    = np.array([_pose_to_T(*gt_poses[j]) for j in matched[valid]])
        print(f"Loaded {len(self._files)} frames.")

    def __len__(self):
        return max(0, len(self._files) - self.stride)

    def _load_img(self, idx):
        img = cv2.imread(str(self.root / self._files[idx]))
        assert img is not None

        if self.undistort:
            img = cv2.undistort(img, self.K, self.params['dist'])
        return img

    def get_relative_pose(self, i, j):
        T_rel = np.linalg.inv(self._Ts[i]) @ self._Ts[j]
        return T_rel[:3, :3], T_rel[:3, 3]

    def __getitem__(self, idx):
        assert 0 <= idx < len(self)
        j = idx + self.stride

        R, t = self.get_relative_pose(idx, j)
        return self._load_img(idx), self._load_img(j), R, t

    def get_pairs(self, n_pairs=20, start=0):
        idxs = np.linspace(start, len(self) - 1, n_pairs, dtype=int)
        pairs = []

        for i in idxs:
            try:
                pairs.append(self[int(i)])
            except Exception as e:
                print(f"  Skipping {i}: {e}")
        return pairs

    def get_K(self):
        return self.K.copy()
