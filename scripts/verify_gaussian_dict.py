"""
Verify numerical stability of Gaussian kernel dictionaries.
Checks determinants and condition numbers of covariance matrices.
"""

import torch
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import os


def compute_covariance_properties(cho1, cho2, cho3, scale=2.0, div_factor=4.0):
    """Compute properties for covariance matrices generated from Cholesky parameters.

    Args:
        cho1, cho2, cho3: arrays of Cholesky parameters
        scale: scaling factor (default 2.0)
        div_factor: division factor (default 4.0, corresponds to para_/4)

    Returns:
        stats dict, determinants tensor, condition_numbers tensor, weighted_cholesky tensor
    """
    gau_dict = torch.tensor(list(product(cho1, cho2, cho3)), dtype=torch.float32)

    # weighted_cholesky = (param / div_factor) * scale
    weighted = gau_dict / div_factor * scale

    a = weighted[:, 0]
    b = weighted[:, 1]
    c = weighted[:, 2]

    # determinants: det(Sigma) = a^2 c^2 - a^2 b^2 = a^2 (c^2 - b^2)
    determinants = a**2 * c**2 - (a * b)**2

    # eigenvalue-based condition number estimate
    trace = a**2 + b**2 + c**2
    delta = (a**2 + b**2 + c**2)**2 - 4 * (a**2 * c**2 - (a * b)**2)
    delta = torch.clamp(delta, min=0.0)

    lambda_max = (trace + torch.sqrt(delta)) / 2
    lambda_min = (trace - torch.sqrt(delta)) / 2

    condition_numbers = lambda_max / (lambda_min + 1e-10)

    stats = {
        'num_gaussians': len(gau_dict),
        'determinants': {
            'min': determinants.min().item(),
            'max': determinants.max().item(),
            'mean': determinants.mean().item(),
            'negative_count': (determinants < 0).sum().item(),
            'near_zero_count': (torch.abs(determinants) < 1e-6).sum().item(),
        },
        'condition_numbers': {
            'min': condition_numbers.min().item(),
            'max': condition_numbers.max().item(),
            'mean': condition_numbers.mean().item(),
            'median': condition_numbers.median().item(),
            'high_condition_count': (condition_numbers > 1000).sum().item(),
        },
        'weighted_cholesky': {
            'a_range': (a.min().item(), a.max().item()),
            'b_range': (b.min().item(), b.max().item()),
            'c_range': (c.min().item(), c.max().item()),
        },
        'constraint_violation': {
            # count of violations where |c| <= |b|
            'count': (torch.abs(c) <= torch.abs(b)).sum().item(),
            'percentage': (torch.abs(c) <= torch.abs(b)).float().mean().item() * 100,
        }
    }

    return stats, determinants, condition_numbers, weighted


def visualize_properties(det1, cond1, det2, cond2, save_path='./'):
    """Visualize comparison between two dictionaries' properties."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Determinant distributions
    ax = axes[0, 0]
    ax.hist(det1.numpy(), bins=50, alpha=0.6, label='orig dict', color='blue', edgecolor='black')
    ax.hist(det2.numpy(), bins=50, alpha=0.6, label='new dict (5T MRI)', color='red', edgecolor='black')
    ax.axvline(0, color='black', linestyle='--', linewidth=2, label='zero')
    ax.set_xlabel('Determinant value', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Determinant distribution of covariance matrices', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Log10 determinant (positive values)
    ax = axes[0, 1]
    det1_pos = det1[det1 > 0]
    det2_pos = det2[det2 > 0]
    if len(det1_pos) > 0:
        ax.hist(torch.log10(det1_pos + 1e-10).numpy(), bins=50, alpha=0.6, label='orig dict', color='blue')
    if len(det2_pos) > 0:
        ax.hist(torch.log10(det2_pos + 1e-10).numpy(), bins=50, alpha=0.6, label='new dict (5T MRI)', color='red')
    ax.set_xlabel('log10(Determinant)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Log10 determinant distribution (positive values)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Condition number distribution
    ax = axes[1, 0]
    ax.hist(cond1.numpy(), bins=50, alpha=0.6, label='orig dict', color='blue', range=(0, 1000))
    ax.hist(cond2.numpy(), bins=50, alpha=0.6, label='new dict (5T MRI)', color='red', range=(0, 1000))
    ax.set_xlabel('Condition number', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Condition number distribution (<1000)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Log condition number distribution
    ax = axes[1, 1]
    ax.hist(torch.log10(cond1 + 1).numpy(), bins=50, alpha=0.6, label='orig dict', color='blue')
    ax.hist(torch.log10(cond2 + 1).numpy(), bins=50, alpha=0.6, label='new dict (5T MRI)', color='red')
    ax.set_xlabel('log10(Condition number + 1)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Log condition number distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'gaussian_dict_comparison.png'), dpi=150, bbox_inches='tight')
    print(f"Plots saved to: {os.path.join(save_path, 'gaussian_dict_comparison.png')}")
    plt.close()


def find_problematic_combinations(cho1, cho2, cho3, scale=2.0, div_factor=4.0):
    """Find specific parameter combinations that lead to numerical issues."""
    gau_dict = torch.tensor(list(product(cho1, cho2, cho3)), dtype=torch.float32)
    weighted = gau_dict / div_factor * scale

    a = weighted[:, 0]
    b = weighted[:, 1]
    c = weighted[:, 2]

    determinants = a**2 * c**2 - (a * b)**2

    negative_mask = determinants < 0
    near_zero_mask = torch.abs(determinants) < 1e-6

    problematic = []

    if negative_mask.any():
        neg_indices = torch.where(negative_mask)[0]
        for idx in neg_indices[:10]:
            orig = gau_dict[idx]
            w = weighted[idx]
            det = determinants[idx]
            problematic.append({
                'type': 'negative_det',
                'original': orig.tolist(),
                'weighted': w.tolist(),
                'determinant': det.item(),
            })

    if near_zero_mask.any():
        zero_indices = torch.where(near_zero_mask)[0]
        for idx in zero_indices[:10]:
            orig = gau_dict[idx]
            w = weighted[idx]
            det = determinants[idx]
            problematic.append({
                'type': 'near_zero_det',
                'original': orig.tolist(),
                'weighted': w.tolist(),
                'determinant': det.item(),
            })

    return problematic


def main():
    print("=" * 80)
    print("Gaussian kernel dictionary numerical stability check")
    print("=" * 80)

    print("\nDefining dictionaries...")

    # Original dictionary (natural images)
    cho1_orig = [0, 0.41, 0.62, 0.98, 1.13, 1.29, 1.64, 1.85, 2.36]
    cho2_orig = [-0.86, -0.36, -0.16, 0.19, 0.34, 0.49, 0.84, 1.04, 1.54]
    cho3_orig = [0, 0.33, 0.53, 0.88, 1.03, 1.18, 1.53, 1.73, 2.23]

    # New dictionary (5T MRI)
    cho1_new = [0.56, 0.77, 0.98, 1.19, 1.40, 1.62, 1.83, 2.04, 2.25]
    cho2_new = [-0.42, -0.24, -0.07, 0.11, 0.29, 0.47, 0.65, 0.83, 1.01]
    cho3_new = [0.54, 0.75, 0.97, 1.19, 1.40, 1.62, 1.83, 2.05, 2.26]

    # Test multiple scale values
    scales_to_test = [1.0, 2.0, 4.0]

    for scale in scales_to_test:
        print(f"\n{'='*80}")
        print(f"Testing scale = {scale}")
        print(f"{'='*80}")

        # Compute properties for original dictionary
        print("\n[1] Original dictionary (natural images):")
        stats1, det1, cond1, weighted1 = compute_covariance_properties(
            cho1_orig, cho2_orig, cho3_orig, scale=scale
        )

        print(f"  Total Gaussians: {stats1['num_gaussians']}")
        print("  Determinant stats:")
        print(f"    Range: [{stats1['determinants']['min']:.6f}, {stats1['determinants']['max']:.6f}]")
        print(f"    Mean: {stats1['determinants']['mean']:.6f}")
        print(f"    Negative count: {stats1['determinants']['negative_count']} ({stats1['determinants']['negative_count']/stats1['num_gaussians']*100:.2f}%)")
        print(f"    Near-zero (<1e-6): {stats1['determinants']['near_zero_count']} ({stats1['determinants']['near_zero_count']/stats1['num_gaussians']*100:.2f}%)")

        print("  Condition number stats:")
        print(f"    Range: [{stats1['condition_numbers']['min']:.2f}, {stats1['condition_numbers']['max']:.2f}]")
        print(f"    Median: {stats1['condition_numbers']['median']:.2f}")
        print(f"    High condition count (>1000): {stats1['condition_numbers']['high_condition_count']} ({stats1['condition_numbers']['high_condition_count']/stats1['num_gaussians']*100:.2f}%)")

        print("  Constraint violation (|c| <= |b|):")
        print(f"    Count: {stats1['constraint_violation']['count']} ({stats1['constraint_violation']['percentage']:.2f}%)")

        # Compute properties for new dictionary
        print("\n[2] New dictionary (5T MRI):")
        stats2, det2, cond2, weighted2 = compute_covariance_properties(
            cho1_new, cho2_new, cho3_new, scale=scale
        )

        print(f"  Total Gaussians: {stats2['num_gaussians']}")
        print("  Determinant stats:")
        print(f"    Range: [{stats2['determinants']['min']:.6f}, {stats2['determinants']['max']:.6f}]")
        print(f"    Mean: {stats2['determinants']['mean']:.6f}")
        print(f"    Negative count: {stats2['determinants']['negative_count']} ({stats2['determinants']['negative_count']/stats2['num_gaussians']*100:.2f}%)")
        print(f"    Near-zero (<1e-6): {stats2['determinants']['near_zero_count']} ({stats2['determinants']['near_zero_count']/stats2['num_gaussians']*100:.2f}%)")

        print("  Condition number stats:")
        print(f"    Range: [{stats2['condition_numbers']['min']:.2f}, {stats2['condition_numbers']['max']:.2f}]")
        print(f"    Median: {stats2['condition_numbers']['median']:.2f}")
        print(f"    High condition count (>1000): {stats2['condition_numbers']['high_condition_count']} ({stats2['condition_numbers']['high_condition_count']/stats2['num_gaussians']*100:.2f}%)")

        print("  Constraint violation (|c| <= |b|):")
        print(f"    Count: {stats2['constraint_violation']['count']} ({stats2['constraint_violation']['percentage']:.2f}%)")

        # Comparison analysis
        print("\n[3] Comparison analysis:")
        if stats2['determinants']['negative_count'] > stats1['determinants']['negative_count']:
            print(f"  ⚠️  New dict has more negative-determinant combinations (+{stats2['determinants']['negative_count'] - stats1['determinants']['negative_count']})")

        if stats2['constraint_violation']['count'] > stats1['constraint_violation']['count']:
            print(f"  ⚠️  New dict has more constraint violations (+{stats2['constraint_violation']['count'] - stats1['constraint_violation']['count']})")

        if stats2['condition_numbers']['median'] > stats1['condition_numbers']['median'] * 2:
            print(f"  ⚠️  New dict median condition number significantly higher ({stats2['condition_numbers']['median']:.2f} vs {stats1['condition_numbers']['median']:.2f})")

        # Find problematic combinations
        if stats2['determinants']['negative_count'] > 0 or stats2['determinants']['near_zero_count'] > 10:
            print("\n[4] Example problematic combinations (new dict):")
            problematic = find_problematic_combinations(cho1_new, cho2_new, cho3_new, scale=scale)
            for i, p in enumerate(problematic[:5], 1):
                print(f"  {i}. type: {p['type']}")
                print(f"     original: {p['original']}")
                print(f"     weighted: {p['weighted']}")
                print(f"     determinant: {p['determinant']:.8f}")

    # Generate visualization (use scale=2.0)
    print(f"\n{'='*80}")
    print("Generating visualization (scale=2.0)...")
    print(f"{'='*80}")
    stats1, det1, cond1, _ = compute_covariance_properties(cho1_orig, cho2_orig, cho3_orig, scale=2.0)
    stats2, det2, cond2, _ = compute_covariance_properties(cho1_new, cho2_new, cho3_new, scale=2.0)

    save_path = './scripts/'
    os.makedirs(save_path, exist_ok=True)
    visualize_properties(det1, cond1, det2, cond2, save_path=save_path)

    print("\n" + "="*80)
    print("Conclusions:")
    print("="*80)
    if stats2['determinants']['negative_count'] > 0:
        print("❌ New dict contains negative determinants, leading to invalid covariance matrices!")
        print("   This can directly cause training collapse.")
    elif stats2['determinants']['near_zero_count'] > stats1['determinants']['near_zero_count'] * 2:
        print("⚠️  New dict has many near-zero determinants; numerical instability suspected.")
    else:
        print("✓ Determinants in the new dict look acceptable.")

    if stats2['constraint_violation']['percentage'] > 10:
        print(f"⚠️  {stats2['constraint_violation']['percentage']:.1f}% of combinations violate the |c|>|b| constraint.")

    print("\nRecommendations:")
    if stats2['determinants']['negative_count'] > 0:
        print("1. Narrow cho2 range so that |cho2| < min(cho1, cho3) after scaling.")
        print("2. Or increase the minimum values of cho1 and cho3.")
        print("3. Or add clamping protections for weighted_cholesky in code.")


if __name__ == "__main__":
    main()
