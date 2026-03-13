import numpy as np
import matplotlib.pyplot as plt
import cv2


def _rgb(img):
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def plot_image_pair(img1, img2, title='', figsize=(12, 5)):
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, im, lbl in zip(axes, [img1, img2], ['Image 1', 'Image 2']):
        ax.imshow(_rgb(im)); ax.set_title(lbl); ax.axis('off')

    if title:
        fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    return fig


def plot_degradation_gallery(img, deg_types, severities=(1, 3, 5)):
    from .degradation import apply_degradation, DEGRADATION_LABELS, get_severity_label

    nrows, ncols = len(deg_types) + 1, len(severities)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.5*nrows))

    for j in range(ncols):
        axes[0, j].imshow(_rgb(img))
        axes[0, j].set_title(f'Original (sev={severities[j]})', fontsize=10)
        axes[0, j].axis('off')

    for i, deg in enumerate(deg_types):
        for j, sev in enumerate(severities):
            axes[i+1, j].imshow(_rgb(apply_degradation(img, deg, sev)))
            axes[i+1, j].set_title(f"{DEGRADATION_LABELS[deg]}\n{get_severity_label(deg, sev)}", fontsize=9)
            axes[i+1, j].axis('off')

    plt.tight_layout()
    return fig


def plot_preprocessing_gallery(img, methods):
    from .preprocessing import apply_preprocessing, PREPROCESSING_LABELS

    n = len(methods)
    ncols = min(n, 4)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.5*nrows))
    axes = np.array(axes).ravel()

    for ax, m in zip(axes, methods):
        ax.imshow(_rgb(apply_preprocessing(img, m)))
        ax.set_title(PREPROCESSING_LABELS.get(m, m), fontsize=10)
        ax.axis('off')
    for ax in axes[n:]:
        ax.axis('off')

    plt.tight_layout()
    return fig


def plot_matches(img1, img2, p1, p2, mask=None, max_n=60, title=''):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    canvas = np.zeros((max(h1, h2), w1+w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = _rgb(img1)
    canvas[:h2, w1:] = _rgb(img2)

    n = len(p1)
    mask = np.ones(n, dtype=bool) if mask is None else mask[:n]
    idx = np.where(mask)[0]
    if len(idx) > max_n:
        idx = idx[np.random.choice(len(idx), max_n, replace=False)]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.imshow(canvas)
    for i in idx:
        x1, y1 = p1[i]
        x2, y2 = p2[i][0]+w1, p2[i][1]
        c = 'lime' if mask[i] else 'red'
        ax.plot([x1, x2], [y1, y2], color=c, lw=0.6, alpha=0.7)
        ax.plot(x1, y1, 'o', color=c, ms=2)
        ax.plot(x2, y2, 'o', color=c, ms=2)

    nin = mask.sum()
    ax.set_title(f"{title} | Matches: {n}  Inliers: {nin}  Ratio: {nin/max(n,1):.2f}", fontsize=11)
    ax.axis('off')
    plt.tight_layout()
    return fig


def plot_metric_vs_severity(res_list, deg_type, metric,
                             models=('essential',), preprocess='none',
                             ylabel=None, title=None):
    from .degradation import DEGRADATION_LABELS

    fig, ax = plt.subplots(figsize=(7, 4))
    sevs = sorted({r['severity'] for r in res_list})

    for mdl in models:
        ms, ss = [], []
        for sev in sevs:
            vals = [r[metric] for r in res_list
                    if r['deg_type']==deg_type and r['severity']==sev
                    and r['model']==mdl and r['preprocessing']==preprocess
                    and not np.isnan(r[metric])]
            ms.append(np.mean(vals) if vals else np.nan)
            ss.append(np.std(vals)  if vals else 0.0)
        ms, ss = np.array(ms), np.array(ss)
        ax.plot(sevs, ms, 'o-', label=f"{mdl.capitalize()} matrix", lw=2)
        ax.fill_between(sevs, ms-ss, ms+ss, alpha=0.15)

    ax.set_xlabel('Severity', fontsize=12)
    ax.set_ylabel(ylabel or metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title(title or f"{ylabel or metric} vs. Severity\n({DEGRADATION_LABELS.get(deg_type, deg_type)})", fontsize=12)
    ax.set_xticks(sevs)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_preprocessing_comparison(res_list, deg_type, severity, metric,
                                   model='essential', ylabel=None):
    from .preprocessing import PREPROCESSING_LABELS
    from .degradation import DEGRADATION_LABELS

    methods = sorted({r['preprocessing'] for r in res_list})
    ms, ss = [], []
    for m in methods:
        vals = [r[metric] for r in res_list
                if r['deg_type']==deg_type and r['severity']==severity
                and r['preprocessing']==m and r['model']==model
                and not np.isnan(r[metric])]
        ms.append(np.mean(vals) if vals else np.nan)
        ss.append(np.std(vals)  if vals else 0.0)

    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x, ms, yerr=ss, capsize=5, alpha=0.8,
           color=plt.cm.tab10(np.linspace(0, 1, len(methods))))

    ax.set_xticks(x)
    ax.set_xticklabels([PREPROCESSING_LABELS.get(m, m) for m in methods], rotation=30, ha='right', fontsize=10)
    ax.set_ylabel(ylabel or metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f"Preprocessing — {DEGRADATION_LABELS.get(deg_type, deg_type)} (sev={severity}, {model})", fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    return fig


def plot_degradation_overview(res_list, metric, model='essential',
                               preprocess='none', ylabel=None):
    from .degradation import DEGRADATION_FUNCTIONS, DEGRADATION_LABELS

    deg_types = list(DEGRADATION_FUNCTIONS.keys())
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.ravel()
    sevs = sorted({r['severity'] for r in res_list})

    for ax, dt in zip(axes, deg_types):
        ms, ss = [], []
        for sev in sevs:
            vals = [r[metric] for r in res_list
                    if r['deg_type']==dt and r['severity']==sev
                    and r['model']==model and r['preprocessing']==preprocess
                    and not np.isnan(r[metric])]
            ms.append(np.mean(vals) if vals else np.nan)
            ss.append(np.std(vals)  if vals else 0.0)
        ms, ss = np.array(ms), np.array(ss)

        ax.plot(sevs, ms, 'o-', lw=2, color='steelblue')
        ax.fill_between(sevs, ms-ss, ms+ss, alpha=0.15, color='steelblue')
        ax.set_title(DEGRADATION_LABELS[dt], fontsize=12)
        ax.set_xlabel('Severity')
        ax.set_ylabel(ylabel or metric.replace('_', ' ').title())
        ax.set_xticks(sevs)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{ylabel or metric.replace('_', ' ').title()} vs. Severity ({model}, no preprocessing)",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    return fig


def plot_model_comparison(res_list, deg_type, metric, preprocess='none', ylabel=None):
    from .degradation import DEGRADATION_LABELS

    sevs = sorted({r['severity'] for r in res_list})
    fig, ax = plt.subplots(figsize=(7, 4))

    for mdl, color in [('essential', 'steelblue'), ('homography', 'darkorange')]:
        ms = []
        for sev in sevs:
            vals = [r[metric] for r in res_list
                    if r['deg_type']==deg_type and r['severity']==sev
                    and r['model']==mdl and r['preprocessing']==preprocess
                    and not np.isnan(r[metric])]
            ms.append(np.mean(vals) if vals else np.nan)
        ax.plot(sevs, ms, 'o-', label=mdl.capitalize(), color=color, lw=2)

    ax.set_xlabel('Severity', fontsize=12)
    ax.set_ylabel(ylabel or metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f"Essential vs. Homography — {DEGRADATION_LABELS.get(deg_type, deg_type)}", fontsize=12)
    ax.set_xticks(sevs)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def save_fig(fig, path, dpi=150):
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f"Saved: {path}")
