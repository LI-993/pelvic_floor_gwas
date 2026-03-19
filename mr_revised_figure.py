"""
Mendelian Randomization - Revised Figure for Manuscript
Panel A: Forest plot (IVW + sensitivity methods, excluding positive controls)
Panel B: Clean causal network diagram
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe

# ── Load data ──
df = pd.read_csv(r'D:\Nproject\gwas\pelvic_floor_gwas\results\mr\mr_bidirectional_results.csv')

# ── Color palette (matching revised figures) ──
colors = {
    'POP': '#7B68AE',       # purple (from Fig2)
    'BPH': '#E8A838',       # orange
    'Bladder': '#2D8E4E',   # green
    'Constip.': '#C47A6E',  # brown/salmon
    'F.Prolapse': '#5B9BD5', # blue
    'Incontin.': '#8B6BAE',  # violet
}

# Full names for display
name_map = {
    'POP': 'POP',
    'BPH': 'BPH',
    'FemaleProlapse': 'F.Prolapse',
    'Incontinence': 'Incontin.',
    'Constipation': 'Constip.',
    'Bladder': 'Bladder'
}

# ── Filter: exclude POP<->FemaleProlapse (positive control, essentially same trait) ──
mask = ~(
    ((df['exposure'] == 'POP') & (df['outcome'] == 'FemaleProlapse')) |
    ((df['exposure'] == 'FemaleProlapse') & (df['outcome'] == 'POP'))
)
df_filtered = df[mask].copy()

# Significant results for forest plot (IVW p < 0.05)
df_sig = df_filtered[df_filtered['ivw_p'] < 0.05].copy()
df_sig['label'] = df_sig.apply(
    lambda r: f"{name_map[r['exposure']]} → {name_map[r['outcome']]}", axis=1
)
df_sig = df_sig.sort_values('ivw_beta', ascending=True).reset_index(drop=True)

# ── Figure setup ──
fig = plt.figure(figsize=(16, 7))
gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1], wspace=0.35)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

# ═══════════════════════════════════════════
# Panel A: Forest Plot with multi-method comparison
# ═══════════════════════════════════════════
y_pos = np.arange(len(df_sig))
bar_height = 0.25

methods = [
    ('ivw_beta', 'ivw_se', 'IVW', '#2D5F8A', 'o', 8),
    ('wm_beta', 'wm_se', 'Weighted Median', '#E8A838', 's', 7),
    ('egger_beta', 'egger_se', 'MR-Egger', '#C47A6E', 'D', 7),
]

for j, (beta_col, se_col, method_name, color, marker, ms) in enumerate(methods):
    offset = (j - 1) * bar_height
    betas = df_sig[beta_col].values
    ses = df_sig[se_col].values
    ci_lo = betas - 1.96 * ses
    ci_hi = betas + 1.96 * ses

    ax1.errorbar(betas, y_pos + offset, xerr=1.96 * ses,
                 fmt=marker, color=color, markersize=ms, capsize=3,
                 linewidth=1.5, markeredgecolor='white', markeredgewidth=0.5,
                 label=method_name, zorder=3)

# Add significance annotations (IVW p-value)
for i, row in df_sig.iterrows():
    idx = df_sig.index.get_loc(i)
    p = row['ivw_p']
    n = int(row['n_snps'])
    if p < 0.001:
        sig_text = f"P={p:.1e}  (n={n})"
    else:
        sig_text = f"P={p:.3f}  (n={n})"
    ax1.text(max(row['ivw_beta'] + 1.96 * row['ivw_se'] + 0.01, 0.42), idx,
             sig_text, va='center', fontsize=8, color='#333333')

# Null line
ax1.axvline(x=0, color='grey', linestyle='--', linewidth=0.8, alpha=0.7)

ax1.set_yticks(y_pos)
ax1.set_yticklabels(df_sig['label'].values, fontsize=10)
ax1.set_xlabel('Causal Effect (β) with 95% CI', fontsize=11)
ax1.set_title('A  Mendelian Randomization Forest Plot', fontsize=13, fontweight='bold', loc='left')
ax1.legend(loc='lower right', fontsize=9, framealpha=0.9)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_xlim(-0.15, 0.65)

# ═══════════════════════════════════════════
# Panel B: Clean Causal Network
# ═══════════════════════════════════════════
# Node positions (manually placed for clarity)
node_pos = {
    'POP':        (0.15, 0.80),
    'F.Prolapse': (0.15, 0.20),
    'BPH':        (0.85, 0.80),
    'Incontin.':  (0.85, 0.20),
    'Constip.':   (0.50, -0.10),
}

# Draw nodes
for node, (x, y) in node_pos.items():
    circle = plt.Circle((x, y), 0.09, color=colors[node], alpha=0.85, zorder=5)
    ax2.add_patch(circle)
    ax2.text(x, y, node, ha='center', va='center', fontsize=9, fontweight='bold',
             color='white', zorder=6,
             path_effects=[pe.withStroke(linewidth=1, foreground='#333333')])

# Significant causal edges (IVW p < 0.05, excluding positive controls)
edges = []
for _, row in df_sig.iterrows():
    exp = name_map[row['exposure']]
    out = name_map[row['outcome']]
    edges.append({
        'from': exp, 'to': out,
        'beta': row['ivw_beta'],
        'p': row['ivw_p'],
        'n_snps': int(row['n_snps']),
    })

# Draw edges with arrows
for edge in edges:
    if edge['from'] not in node_pos or edge['to'] not in node_pos:
        continue
    x1, y1 = node_pos[edge['from']]
    x2, y2 = node_pos[edge['to']]

    # Direction vector
    dx, dy = x2 - x1, y2 - y1
    dist = np.sqrt(dx**2 + dy**2)
    ux, uy = dx/dist, dy/dist

    # Shorten to avoid overlap with circles
    r = 0.10
    sx, sy = x1 + ux * r, y1 + uy * r
    ex, ey = x2 - ux * r, y2 - uy * r

    # Check if bidirectional - offset curves
    reverse_exists = any(
        e['from'] == edge['to'] and e['to'] == edge['from'] for e in edges
    )

    # Line properties based on significance
    if edge['p'] < 0.001:
        lw = 2.5
        alpha = 0.9
    elif edge['p'] < 0.01:
        lw = 2.0
        alpha = 0.8
    else:
        lw = 1.5
        alpha = 0.65

    edge_color = '#D94444' if edge['beta'] > 0 else '#4488CC'

    if reverse_exists:
        # Curve the arrow slightly
        mid_x = (sx + ex) / 2
        mid_y = (sy + ey) / 2
        # Perpendicular offset
        perp_x, perp_y = -uy * 0.04, ux * 0.04

        style = f"arc3,rad=0.2"
        arrow = FancyArrowPatch(
            (sx, sy), (ex, ey),
            connectionstyle=style,
            arrowstyle='->', mutation_scale=15,
            color=edge_color, linewidth=lw, alpha=alpha, zorder=3
        )
    else:
        arrow = FancyArrowPatch(
            (sx, sy), (ex, ey),
            arrowstyle='->', mutation_scale=15,
            color=edge_color, linewidth=lw, alpha=alpha, zorder=3
        )
    ax2.add_patch(arrow)

    # Label: beta value
    mid_x = (sx + ex) / 2
    mid_y = (sy + ey) / 2
    # Offset label perpendicular to edge
    offset_x = -uy * 0.05
    offset_y = ux * 0.05
    if reverse_exists:
        offset_x *= 1.8
        offset_y *= 1.8

    beta_text = f"β={edge['beta']:.2f}"
    sig_star = '***' if edge['p'] < 0.001 else ('**' if edge['p'] < 0.01 else '*')
    ax2.text(mid_x + offset_x, mid_y + offset_y, f"{beta_text}{sig_star}",
             ha='center', va='center', fontsize=7.5, color=edge_color,
             fontweight='bold', zorder=7,
             bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                       edgecolor='none', alpha=0.85))

# Legend for edge colors
pos_patch = mpatches.Patch(color='#D94444', label='Positive effect (risk)')
neg_patch = mpatches.Patch(color='#4488CC', label='Negative effect (protective)')
sig_text = ax2.text(0.02, -0.05, '*** P<0.001  ** P<0.01  * P<0.05',
                     fontsize=8, color='#555555', transform=ax2.transAxes)
ax2.legend(handles=[pos_patch, neg_patch], loc='upper left', fontsize=8,
           framealpha=0.9, bbox_to_anchor=(-0.02, 1.02))

ax2.set_xlim(-0.10, 1.10)
ax2.set_ylim(-0.35, 1.05)
ax2.set_aspect('equal')
ax2.axis('off')
ax2.set_title('B  Causal Network (IVW, P<0.05)', fontsize=13, fontweight='bold', loc='left')

plt.tight_layout()

# Save
out_dir = r'D:\Nproject\gwas\pelvic_floor_gwas\figures\revised'
fig.savefig(f'{out_dir}/Fig_MR_revised.png', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(f'{out_dir}/Fig_MR_revised.pdf', bbox_inches='tight', facecolor='white')
print("Saved to figures/revised/Fig_MR_revised.png and .pdf")

# ── Print summary table ──
print("\n=== MR Results Summary ===")
print(f"{'Exposure':<18} {'Outcome':<18} {'N_SNPs':>6} {'IVW_β':>8} {'IVW_P':>12} {'WM_β':>8} {'WM_P':>12} {'Egger_β':>8} {'Egger_P':>12} {'Egger_int_P':>12}")
print("-" * 130)
for _, r in df_filtered.sort_values('ivw_p').iterrows():
    sig = '***' if r['ivw_p'] < 0.001 else ('**' if r['ivw_p'] < 0.01 else ('*' if r['ivw_p'] < 0.05 else ''))
    print(f"{r['exposure']:<18} {r['outcome']:<18} {int(r['n_snps']):>6} {r['ivw_beta']:>8.3f} {r['ivw_p']:>12.2e}{sig:<3} {r['wm_beta']:>8.3f} {r['wm_p']:>12.2e} {r['egger_beta']:>8.3f} {r['egger_p']:>12.2e} {r['egger_intercept_p']:>12.3f}")
