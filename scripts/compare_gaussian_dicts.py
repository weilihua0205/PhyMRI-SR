#!/usr/bin/env python3
"""
compare_gaussian_dicts.py

Generate and compare two Gaussian dictionaries (annotated preset vs gaussian_std generated).
Outputs: CSV metrics, histograms, scatter plots, and comparison images for example kernels.

Usage:
    python scripts/compare_gaussian_dicts.py

Runs standalone without modifying source code.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import csv


def make_dict(cho1, cho2, cho3):
    """Return a dictionary array with shape (N,3) and append a zero row at the end (consistent with original code)."""
    combos = np.array(list(product(cho1, cho2, cho3)), dtype=float)
    combos = np.vstack([combos, np.zeros((1,3), dtype=float)])
    return combos


def cholesky_to_cov(cho):
    """Convert a single Cholesky triple (a,b,c) to a 2x2 covariance matrix.

    Assumes L = [[a, 0], [b, c]] and Sigma = L @ L.T
    """
    a, b, c = cho
    L = np.array([[a, 0.0], [b, c]], dtype=float)
    return L @ L.T


def cov_metrics(S):
    """Compute several scalar metrics for a covariance matrix: eigenvalues (descending), determinant, trace, principal axis angle (radians)."""
    vals, vecs = np.linalg.eigh(S)
    # eigh returns ascending eigenvalues; reverse to descending
    vals = vals[::-1]
    vecs = vecs[:, ::-1]
    eig1, eig2 = vals[0], vals[1]
    det = np.linalg.det(S)
    tr = np.trace(S)
    # principal axis angle: angle of eigenvector corresponding to largest eigenvalue
    v = vecs[:, 0]
    angle = np.arctan2(v[1], v[0])
    return {
        'eig1': float(eig1),
        'eig2': float(eig2),
        'det': float(det),
        'trace': float(tr),
        'angle': float(angle),
    }


def gaussian_kernel_from_cov(S, grid_extent=3.0, size=101, det_thresh=1e-12, eig_thresh=1e-12):
        """Generate a normalized 2D Gaussian kernel (zero-mean) centered in the grid.

        If the covariance matrix is near-singular (determinant or max eigenvalue below thresholds),
        return a delta kernel (center=1, others=0) to handle zero-covariance cases gracefully.

        Parameters:
            S: 2x2 covariance matrix
            grid_extent: half-width of the grid (x,y in [-grid_extent, grid_extent])
            size: number of pixels per side
            det_thresh: determinant threshold to consider singular
            eig_thresh: max-eigenvalue threshold to consider singular
        Returns:
            Normalized kernel (max 1), or a delta kernel
        """
    xs = np.linspace(-grid_extent, grid_extent, size)
    ys = np.linspace(-grid_extent, grid_extent, size)
    X, Y = np.meshgrid(xs, ys)
    XY = np.stack([X.ravel(), Y.ravel()], axis=-1)

    # Check for singular cases: determinant or max eigenvalue close to 0
    try:
        det = float(np.linalg.det(S))
    except Exception:
        det = 0.0
    try:
        eigs = np.linalg.eigvalsh(S)
        max_eig = float(np.max(eigs))
    except Exception:
        max_eig = 0.0

    if det < det_thresh or max_eig < eig_thresh:
        K = np.zeros((size, size), dtype=float)
        K[size // 2, size // 2] = 1.0
        return K

    # Normal case: compute Gaussian kernel
    invS = np.linalg.inv(S)
    exponent = -0.5 * np.sum((XY @ invS) * XY, axis=1)
    Z = np.exp(exponent).reshape(size, size)
    # normalize to max 1 for easier visual comparison
    Z = Z / Z.max()
    return Z


def main():
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs_dictionary', 'gaussian_comparison')
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Annotated preset (original annotations)
    cho1_orig = np.array([0.0, 0.41, 0.62, 0.98, 1.13, 1.29, 1.64, 1.85, 2.36], dtype=float)
    cho2_orig = np.array([-0.86, -0.36, -0.16, 0.19, 0.34, 0.49, 0.84, 1.04, 1.54], dtype=float)
    cho3_orig = np.array([0.0, 0.33, 0.53, 0.88, 1.03, 1.18, 1.53, 1.73, 2.23], dtype=float)

    # gaussian_std generated version (currently used by the model)
    cho1_new = np.array([0.56, 0.77, 0.98, 1.19, 1.40, 1.62, 1.83, 2.04, 2.25], dtype=float)
    cho2_new = np.array([-0.42, -0.24, -0.07, 0.11, 0.29, 0.47, 0.65, 0.83, 1.01], dtype=float)
    cho3_new = np.array([0.54, 0.75, 0.97, 1.19, 1.40, 1.62, 1.83, 2.05, 2.26], dtype=float)

    dict_orig = make_dict(cho1_orig, cho2_orig, cho3_orig)
    dict_new = make_dict(cho1_new, cho2_new, cho3_new)

    assert dict_orig.shape == dict_new.shape, "The two dictionaries should have the same length"
    N = dict_orig.shape[0]

    # Compute covariance matrices and metrics for each entry
    metrics_orig = []
    metrics_new = []
    for i in range(N):
        S_o = cholesky_to_cov(dict_orig[i])
        S_n = cholesky_to_cov(dict_new[i])
        metrics_orig.append(cov_metrics(S_o))
        metrics_new.append(cov_metrics(S_n))

    # Convert metrics to numpy arrays for analysis
    keys = ['eig1', 'eig2', 'det', 'trace', 'angle']
    mo = {k: np.array([m[k] for m in metrics_orig]) for k in keys}
    mn = {k: np.array([m[k] for m in metrics_new]) for k in keys}

    # Compute differences
    diff = {k: mn[k] - mo[k] for k in keys}

    # Save metrics to CSV
    csv_path = os.path.join(out_dir, 'gaussian_metrics_comparison.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['index'] + [f'orig_{k}' for k in keys] + [f'new_{k}' for k in keys] + [f'diff_{k}' for k in keys])
        for i in range(N):
            row = [i] + [mo[k][i] for k in keys] + [mn[k][i] for k in keys] + [diff[k][i] for k in keys]
            writer.writerow(row)

    # Plot: histograms of determinant and its difference
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.hist(mo['det'], bins=50, alpha=0.6, label='orig')
    plt.hist(mn['det'], bins=50, alpha=0.6, label='new')
    plt.title('Determinant distributions')
    plt.legend()

    plt.subplot(1,2,2)
    plt.hist(diff['det'], bins=50, color='tab:purple')
    plt.title('Determinant (new - orig)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'determinant_histograms.png'))
    plt.close()

    # Plot: scatter of principal eigenvalue differences
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.scatter(mo['eig1'], mn['eig1'], s=8)
    plt.xlabel('orig eig1')
    plt.ylabel('new eig1')
    plt.title('eig1: orig vs new')
    plt.plot([mo['eig1'].min(), mo['eig1'].max()], [mo['eig1'].min(), mo['eig1'].max()], 'r--')

    plt.subplot(1,2,2)
    plt.scatter(mo['angle'], mn['angle'], s=8)
    plt.xlabel('orig angle')
    plt.ylabel('new angle')
    plt.title('principal axis angle: orig vs new')
    plt.plot([-np.pi, np.pi], [-np.pi, np.pi], 'r--')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'eig1_angle_scatter.png'))
    plt.close()

    # Select representative indices for visualization: min/median/max determinant difference and index 0
    idx_min = int(np.argmin(diff['det']))
    idx_med = int(np.argsort(diff['det'])[len(diff['det'])//2])
    idx_max = int(np.argmax(diff['det']))
    sample_idxs = [0, idx_min, idx_med, idx_max]
    sample_idxs = sorted(set([i for i in sample_idxs if 0 <= i < N]))

    # For each selected index save comparison plots (orig vs new Gaussian kernels and covariance heatmaps)
    for idx in sample_idxs:
        S_o = cholesky_to_cov(dict_orig[idx])
        S_n = cholesky_to_cov(dict_new[idx])
        K_o = gaussian_kernel_from_cov(S_o, grid_extent=3.0, size=161)
        K_n = gaussian_kernel_from_cov(S_n, grid_extent=3.0, size=161)

        fig, axes = plt.subplots(1,3, figsize=(12,4))
        im0 = axes[0].imshow(K_o, cmap='inferno', extent=[-3,3,-3,3])
        axes[0].set_title(f'orig idx={idx}')
        fig.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(K_n, cmap='inferno', extent=[-3,3,-3,3])
        axes[1].set_title(f'new idx={idx}')
        fig.colorbar(im1, ax=axes[1])

        # Difference kernel
        im2 = axes[2].imshow(K_n - K_o, cmap='bwr', vmin=-np.max(np.abs(K_n - K_o)), vmax=np.max(np.abs(K_n - K_o)), extent=[-3,3,-3,3])
        axes[2].set_title('new - orig')
        fig.colorbar(im2, ax=axes[2])

        plt.suptitle(f'Gaussian kernel comparison idx={idx}')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'kernel_compare_idx_{idx}.png'))
        plt.close()

    # Write brief summary statistics
    summary_path = os.path.join(out_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write('Gaussian dict comparison summary\n')
        f.write(f'Total entries: {N}\n')
        for k in ['eig1', 'eig2', 'det', 'trace', 'angle']:
            f.write(f"{k}: mean(orig)={mo[k].mean():.6g}, mean(new)={mn[k].mean():.6g}, mean(diff)={diff[k].mean():.6g}\n")
        f.write('\nRepresentative indices: ' + ','.join(map(str, sample_idxs)) + '\n')

    print('Comparison complete. Outputs saved to:', out_dir)


if __name__ == '__main__':
    main()
