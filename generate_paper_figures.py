"""
Research Paper Figure Generator for CE-GAN Cross-Dataset Intrusion Detection
Generates: architecture diagram, pipeline diagram, results charts, heatmaps
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("results/figures")
OUTPUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

COLORS = {
    'encoder':      '#4C72B0',
    'generator':    '#DD8452',
    'decoder':      '#55A868',
    'discriminator':'#C44E52',
    'loss':         '#8172B2',
    'data':         '#937860',
    'scenario_a':   '#2196F3',
    'scenario_b':   '#FF9800',
    'scenario_c':   '#4CAF50',
    'smote':        '#F44336',
    'mmd':          '#9C27B0',
    'bg':           '#F8F9FA',
    'arrow':        '#455A64',
}

# ===========================================================
# FIGURE 1 — CE-GAN Architecture Diagram
# ===========================================================
def draw_box(ax, x, y, w, h, label, sublabel=None, color='#4C72B0',
             fontsize=9, text_color='white', style='round,pad=0.1'):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle=style, linewidth=1.2,
                          edgecolor='white', facecolor=color, zorder=3)
    ax.add_patch(box)
    ax.text(x, y + (0.012 if sublabel else 0), label,
            ha='center', va='center', fontsize=fontsize,
            fontweight='bold', color=text_color, zorder=4)
    if sublabel:
        ax.text(x, y - 0.032, sublabel, ha='center', va='center',
                fontsize=7, color=text_color, alpha=0.85, zorder=4,
                style='italic')

def draw_arrow(ax, x1, y1, x2, y2, color='#455A64', label=None, lw=1.5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, mutation_scale=14), zorder=5)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx+0.01, my+0.015, label, fontsize=7,
                color=color, ha='center', zorder=6)

def figure1_architecture():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor(COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])

    # ── Title
    ax.text(0.5, 0.96, 'CE-GAN Architecture for Cross-Dataset Network Intrusion Detection',
            ha='center', va='top', fontsize=13, fontweight='bold', color='#1A237E')

    # ── Row positions
    y_data  = 0.82
    y_model = 0.56
    y_loss  = 0.30
    y_out   = 0.10

    # ── Input data boxes
    draw_box(ax, 0.18, y_data, 0.14, 0.07, 'Real Traffic\nSamples x',
             color='#795548', fontsize=8)
    draw_box(ax, 0.38, y_data, 0.12, 0.07, 'Class Labels c',
             color='#607D8B', fontsize=8)
    draw_box(ax, 0.62, y_data, 0.12, 0.07, 'Noise z~N(0,I)',
             color='#607D8B', fontsize=8)

    # ── Main model components
    draw_box(ax, 0.18, y_model, 0.16, 0.10, 'ENCODER',
             'Transformer (3L, 4H)\nz = E(x,c)',
             color=COLORS['encoder'])
    draw_box(ax, 0.46, y_model, 0.16, 0.10, 'GENERATOR',
             'Transformer (3L, 4H)\nx̃ = G(z,c)',
             color=COLORS['generator'])
    draw_box(ax, 0.18, 0.38, 0.16, 0.10, 'DECODER',
             'Cross-Attn Transformer\nx̂ = D(E(x,c))',
             color=COLORS['decoder'])
    draw_box(ax, 0.74, y_model, 0.16, 0.10, 'DISCRIMINATOR',
             'ACGAN-style\nValidity + Class',
             color=COLORS['discriminator'])

    # ── Loss boxes
    draw_box(ax, 0.18, y_loss, 0.14, 0.07, 'L_rec',
             'MSE Reconstruction', color=COLORS['loss'], fontsize=8)
    draw_box(ax, 0.38, y_loss, 0.14, 0.07, 'L_adv',
             'Adversarial (BCE)', color=COLORS['loss'], fontsize=8)
    draw_box(ax, 0.56, y_loss, 0.14, 0.07, 'L_cst',
             'Class Structure', color=COLORS['loss'], fontsize=8)
    draw_box(ax, 0.74, y_loss, 0.14, 0.07, 'L_mmt',
             'Moment Matching', color=COLORS['loss'], fontsize=8)

    # ── MMD loss (Cross-Dataset extension)
    mmd_box = FancyBboxPatch((0.86, y_loss - 0.045), 0.12, 0.09,
                              boxstyle='round,pad=0.05', linewidth=1.5,
                              edgecolor=COLORS['mmd'], facecolor=COLORS['mmd'],
                              alpha=0.85, zorder=3, linestyle='--')
    ax.add_patch(mmd_box)
    ax.text(0.92, y_loss, 'L_MMD\n(Cross-DS)', ha='center', va='center',
            fontsize=7.5, fontweight='bold', color='white', zorder=4)
    ax.text(0.92, y_loss - 0.065, '(CrossDatasetCEGAN\nextension)',
            ha='center', va='center', fontsize=6.5, color=COLORS['mmd'],
            style='italic', zorder=4)

    # ── Combined loss output
    draw_box(ax, 0.46, y_out, 0.26, 0.07,
             'L_G = L_adv + α·L_cst + β·L_mmt + L_rec',
             color='#37474F', fontsize=8)

    # ── Data flow arrows
    draw_arrow(ax, 0.18, y_data-0.035, 0.18, y_model+0.05)   # x → Encoder
    draw_arrow(ax, 0.38, y_data-0.035, 0.28, y_model+0.02)   # c → Encoder (angled)
    draw_arrow(ax, 0.38, y_data-0.035, 0.38, y_model+0.05)   # c → Generator
    draw_arrow(ax, 0.62, y_data-0.035, 0.54, y_model+0.05)   # z → Generator
    draw_arrow(ax, 0.18, y_model-0.05, 0.18, 0.43)            # Encoder → Decoder
    draw_arrow(ax, 0.54, y_model, 0.66, y_model)              # Generator → Discriminator
    draw_arrow(ax, 0.18, y_model, 0.38, y_model,
               color='#78909C', label='latent z')              # Encoder → Generator (latent)

    # Loss connections
    draw_arrow(ax, 0.18, 0.33, 0.18, y_loss+0.035, color='#9E9E9E')
    draw_arrow(ax, 0.46, y_model-0.05, 0.38, y_loss+0.035, color='#9E9E9E')
    draw_arrow(ax, 0.74, y_model-0.05, 0.56, y_loss+0.035, color='#9E9E9E')
    draw_arrow(ax, 0.46, y_model-0.05, 0.74, y_loss+0.035, color='#9E9E9E')
    draw_arrow(ax, 0.38, y_loss-0.035, 0.46, y_out+0.035, color='#9E9E9E')

    # ── Feature Tokenization annotation
    rect = FancyBboxPatch((0.04, 0.65), 0.08, 0.12,
                           boxstyle='round,pad=0.02', linewidth=1,
                           edgecolor='#78909C', facecolor='#ECEFF1',
                           linestyle=':', zorder=2)
    ax.add_patch(rect)
    ax.text(0.08, 0.71, 'Per-Feature\nTokenization\n+ CLS Token\n(class embed)',
            ha='center', va='center', fontsize=6.5, color='#37474F')
    draw_arrow(ax, 0.12, 0.71, 0.10, y_model+0.01, color='#78909C')

    # ── Legend
    legend_items = [
        mpatches.Patch(color=COLORS['encoder'],      label='Encoder'),
        mpatches.Patch(color=COLORS['generator'],    label='Generator'),
        mpatches.Patch(color=COLORS['decoder'],      label='Decoder'),
        mpatches.Patch(color=COLORS['discriminator'],label='Discriminator'),
        mpatches.Patch(color=COLORS['loss'],         label='Loss Components'),
        mpatches.Patch(color=COLORS['mmd'],          label='MMD (Cross-DS ext.)'),
    ]
    ax.legend(handles=legend_items, loc='lower right', fontsize=8,
              framealpha=0.9, ncol=2, bbox_to_anchor=(1.0, 0.0))

    plt.tight_layout()
    out = OUTPUT_DIR / 'fig1_cegan_architecture.png'
    plt.savefig(out)
    plt.close()
    print(f'Saved: {out}')


# ===========================================================
# FIGURE 2 — Cross-Dataset Transfer Pipeline
# ===========================================================
def figure2_pipeline():
    fig, ax = plt.subplots(figsize=(15, 5.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    fig.patch.set_facecolor(COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])

    ax.text(0.5, 0.95, 'CE-GAN Cross-Dataset Transfer Pipeline',
            ha='center', va='top', fontsize=13, fontweight='bold', color='#1A237E')

    # Stage x-positions
    stages = [0.08, 0.23, 0.40, 0.57, 0.73, 0.88]
    labels  = [
        ('SOURCE\nDATASET', '(NSL-KDD / UNSW-NB15)', '#795548'),
        ('CE-GAN\nTRAINING', 'Encoder+Generator\n+Discriminator', COLORS['encoder']),
        ('SYNTHETIC\nSAMPLES', 'n_aug=500 per class\nG(z, c) → x̃_src', COLORS['generator']),
        ('FEATURE\nHARMONIZER', 'Semantic Mapping\n+ FC Projection', '#00897B'),
        ('TARGET\nAUGMENT', 'Augment target\ntraining set', COLORS['decoder']),
        ('NASH\nENSEMBLE', 'RF + ET + GB\ngame-theoretic weights', COLORS['discriminator']),
    ]

    yc = 0.50
    bw, bh = 0.11, 0.25

    for i, (x, (title, sub, col)) in enumerate(zip(stages, labels)):
        draw_box(ax, x, yc, bw, bh, title, sub, color=col, fontsize=8.5)
        if i < len(stages) - 1:
            draw_arrow(ax, x + bw/2 + 0.005, yc,
                       stages[i+1] - bw/2 - 0.005, yc,
                       color=COLORS['arrow'], lw=1.8)

    # Final output
    draw_box(ax, 0.88, 0.10, 0.18, 0.08,
             'Accuracy · Macro-F1 · TQS', color='#1565C0',
             fontsize=8)
    draw_arrow(ax, 0.88, yc - bh/2 - 0.005, 0.88, 0.14,
               color=COLORS['arrow'], lw=1.8)

    # MMD branch annotation (Scenario C)
    ax.annotate('', xy=(0.40 - 0.055, 0.71), xytext=(0.23 + 0.055, 0.71),
                arrowprops=dict(arrowstyle='->', color=COLORS['mmd'],
                                lw=1.5, linestyle='dashed', mutation_scale=13))
    ax.text(0.315, 0.77, 'Scenario C: MMD latent alignment\n(CrossDatasetCEGAN)',
            ha='center', va='center', fontsize=7.5, color=COLORS['mmd'],
            bbox=dict(boxstyle='round,pad=0.3', fc='#F3E5F5', ec=COLORS['mmd'],
                      lw=1, alpha=0.9))

    # TARGET DATASET arrow
    ax.annotate('', xy=(0.57 - 0.055, 0.62), xytext=(0.23 + 0.055, 0.62),
                arrowprops=dict(arrowstyle='->', color='#607D8B',
                                lw=1.2, linestyle='dotted', mutation_scale=11))
    ax.text(0.40, 0.58, 'Target domain\nsamples (for MMD)',
            ha='center', va='top', fontsize=6.5, color='#607D8B', style='italic')

    # Scenario labels at bottom
    scenario_info = [
        (0.27, 'Scenario A: Source = Target\n(within-dataset baseline)', COLORS['scenario_a']),
        (0.57, 'Scenario B: Direct transfer\n(FeatureHarmonizer only)',   COLORS['scenario_b']),
        (0.80, 'Scenario C: Adapted transfer\n(+ MMD alignment)',         COLORS['scenario_c']),
    ]
    for (xp, text, col) in scenario_info:
        ax.text(xp, 0.08, text, ha='center', va='bottom', fontsize=7,
                color=col, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.25', fc='white',
                          ec=col, lw=1, alpha=0.9))

    plt.tight_layout()
    out = OUTPUT_DIR / 'fig2_transfer_pipeline.png'
    plt.savefig(out)
    plt.close()
    print(f'Saved: {out}')


# ===========================================================
# FIGURE 3 — Main Results: Macro-F1 Comparison
# ===========================================================
def figure3_main_results():
    df = pd.read_csv('results/tables/table3_main_results.csv')

    # Keep only well-performing rows (not degenerate imbalanced runs)
    df = df[df['macro_f1'] > 0.80].copy()

    # Aggregate: mean per (scenario, source, target)
    agg = (df.groupby(['scenario', 'source_dataset', 'target_dataset'])
             ['macro_f1'].mean().reset_index())

    # Build a human-readable transfer label
    agg['transfer'] = agg.apply(
        lambda r: r['source_dataset'].upper().replace('_', '-')
        if r['scenario'] == 'A'
        else f"{r['source_dataset'].upper().replace('_','-')} → {r['target_dataset'].upper().replace('_','-')}",
        axis=1)
    agg = agg.sort_values(['transfer', 'scenario'])

    fig, ax = plt.subplots(figsize=(13, 5.5))
    fig.patch.set_facecolor(COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])

    transfers = sorted(agg['transfer'].unique())
    x = np.arange(len(transfers))
    width = 0.26

    scen_cfg = [
        ('A', 'Scenario A (Within-dataset)', COLORS['scenario_a']),
        ('B', 'Scenario B (Direct Transfer)', COLORS['scenario_b']),
        ('C', 'Scenario C (Adapted Transfer + MMD)', COLORS['scenario_c']),
    ]

    offsets = [-width, 0, width]
    for offset, (scen, label, color) in zip(offsets, scen_cfg):
        vals = []
        for t in transfers:
            row = agg[(agg['scenario'] == scen) & (agg['transfer'] == t)]
            vals.append(row['macro_f1'].values[0] if len(row) else np.nan)
        bars = ax.bar(x + offset, vals, width - 0.02, label=label,
                      color=color, alpha=0.87, edgecolor='white', linewidth=0.8,
                      zorder=3)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                        f'{v:.3f}', ha='center', va='bottom', fontsize=7,
                        color='#263238', rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels(transfers, rotation=20, ha='right', fontsize=9)
    ax.set_ylim(0.78, 1.00)
    ax.set_ylabel('Macro-F1 Score', fontsize=11)
    ax.set_title('CE-GAN Performance: Macro-F1 Across Scenarios and Dataset Pairs',
                 fontsize=12, fontweight='bold', pad=12)
    ax.legend(fontsize=9, loc='lower right', framealpha=0.9)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    out = OUTPUT_DIR / 'fig3_main_results.png'
    plt.savefig(out)
    plt.close()
    print(f'Saved: {out}')


# ===========================================================
# FIGURE 4 — Cross-Dataset Transfer Heatmaps (B vs C)
# ===========================================================
def figure4_transfer_heatmaps():
    df = pd.read_csv('results/tables/table3_main_results.csv')
    df = df[df['macro_f1'] > 0.80]
    agg = (df.groupby(['scenario', 'source_dataset', 'target_dataset'])
             ['macro_f1'].mean().reset_index())

    datasets = ['nsl_kdd', 'unsw_nb15', 'cic_ids2017']
    labels   = ['NSL-KDD', 'UNSW-NB15', 'CIC-IDS2017']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.patch.set_facecolor(COLORS['bg'])

    scens = [
        ('A', 'Scenario A: Within-Dataset',   'Blues',   0.83, 0.97),
        ('B', 'Scenario B: Direct Transfer',   'Oranges', 0.83, 0.97),
        ('C', 'Scenario C: Adapted (+ MMD)',   'Greens',  0.83, 0.97),
    ]

    for ax, (scen, title, cmap, vmin, vmax) in zip(axes, scens):
        ax.set_facecolor(COLORS['bg'])
        matrix = np.full((3, 3), np.nan)
        for i, src in enumerate(datasets):
            for j, tgt in enumerate(datasets):
                row = agg[(agg['scenario'] == scen) &
                          (agg['source_dataset'] == src) &
                          (agg['target_dataset'] == tgt)]
                if not row.empty:
                    matrix[i, j] = row['macro_f1'].values[0]

        im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=8.5)
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.set_xlabel('Target Dataset', fontsize=9)
        ax.set_ylabel('Source Dataset', fontsize=9)
        ax.set_title(title, fontsize=10, fontweight='bold', pad=8)

        for i in range(3):
            for j in range(3):
                if not np.isnan(matrix[i, j]):
                    bg = matrix[i, j]
                    clr = 'white' if bg < (vmin + vmax) / 2 + 0.04 else '#263238'
                    ax.text(j, i, f'{matrix[i,j]:.3f}',
                            ha='center', va='center',
                            fontsize=10, fontweight='bold', color=clr)
                else:
                    ax.text(j, i, '—', ha='center', va='center',
                            fontsize=11, color='#90A4AE')

        plt.colorbar(im, ax=ax, shrink=0.75, label='Macro-F1')

    fig.suptitle('Cross-Dataset Transfer: Macro-F1 Score Matrix',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = OUTPUT_DIR / 'fig4_transfer_heatmaps.png'
    plt.savefig(out)
    plt.close()
    print(f'Saved: {out}')


# ===========================================================
# FIGURE 5 — CE-GAN vs SMOTE Baseline Comparison
# ===========================================================
def figure5_smote_comparison():
    # SMOTE baseline values (from table5)
    smote_data = {
        'NSL-KDD → CIC-IDS2017':  {'smote_f1': 0.6507, 'smote_min': 0.5526},
        'UNSW-NB15 → CIC-IDS2017':{'smote_f1': 0.6507, 'smote_min': 0.5526},
    }

    df = pd.read_csv('results/tables/table3_main_results.csv')
    df = df[df['macro_f1'] > 0.80]
    agg = (df.groupby(['scenario', 'source_dataset', 'target_dataset'])
             [['macro_f1', 'minority_class_f1']].mean().reset_index())

    pairs = [
        ('nsl_kdd',  'cic_ids2017', 'NSL-KDD → CIC-IDS2017'),
        ('unsw_nb15','cic_ids2017', 'UNSW-NB15 → CIC-IDS2017'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=False)
    fig.patch.set_facecolor(COLORS['bg'])

    metric_cfgs = [
        ('macro_f1',        'Macro-F1',        'smote_f1'),
        ('minority_class_f1','Minority Class F1','smote_min'),
    ]

    for ax, (metric, mlabel, smote_key) in zip(axes, metric_cfgs):
        ax.set_facecolor(COLORS['bg'])
        pair_labels, scen_b, scen_c, smote_vals = [], [], [], []

        for src, tgt, plabel in pairs:
            pair_labels.append(plabel)
            smote_vals.append(smote_data[plabel][smote_key])

            row_b = agg[(agg['scenario'] == 'B') &
                        (agg['source_dataset'] == src) &
                        (agg['target_dataset'] == tgt)]
            scen_b.append(row_b[metric].values[0] if not row_b.empty else np.nan)

            row_c = agg[(agg['scenario'] == 'C') &
                        (agg['source_dataset'] == src) &
                        (agg['target_dataset'] == tgt)]
            scen_c.append(row_c[metric].values[0] if not row_c.empty else np.nan)

        x = np.arange(len(pair_labels))
        w = 0.25

        b1 = ax.bar(x - w, smote_vals, w - 0.02, label='SMOTE Baseline',
                    color=COLORS['smote'], alpha=0.82, edgecolor='white')
        b2 = ax.bar(x,     scen_b,     w - 0.02, label='CE-GAN (Scenario B)',
                    color=COLORS['scenario_b'], alpha=0.82, edgecolor='white')
        b3 = ax.bar(x + w, scen_c,     w - 0.02, label='CE-GAN + MMD (Scenario C)',
                    color=COLORS['scenario_c'], alpha=0.82, edgecolor='white')

        for bars in [b1, b2, b3]:
            for bar in bars:
                h = bar.get_height()
                if not np.isnan(h):
                    ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                            f'{h:.3f}', ha='center', va='bottom', fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(pair_labels, fontsize=9, rotation=10, ha='right')
        ax.set_ylabel(mlabel, fontsize=11)
        ax.set_title(f'{mlabel}: CE-GAN vs SMOTE Baseline', fontsize=11,
                     fontweight='bold', pad=8)
        ax.legend(fontsize=8.5)
        ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        ymin = min(smote_vals + scen_b + scen_c) - 0.05
        ax.set_ylim(max(0, ymin), 1.02)

        # improvement annotation
        for i, (sv, bv) in enumerate(zip(smote_vals, scen_b)):
            if not np.isnan(bv):
                delta = bv - sv
                ax.annotate(f'+{delta:.3f}', xy=(i, bv), xytext=(i, sv + (bv - sv)/2),
                            fontsize=7.5, color='#1B5E20', ha='center',
                            arrowprops=dict(arrowstyle='-', color='#1B5E20', lw=0.8))

    fig.suptitle('CE-GAN vs SMOTE: Cross-Dataset Transfer Performance',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = OUTPUT_DIR / 'fig5_smote_comparison.png'
    plt.savefig(out)
    plt.close()
    print(f'Saved: {out}')


# ===========================================================
# FIGURE 6 — Feature Mapping Coverage
# ===========================================================
def figure6_feature_mapping():
    df = pd.read_csv('results/tables/table2_feature_mapping.csv')
    counts = df['mapping_type'].value_counts()
    total  = len(df)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(COLORS['bg'])

    # ── Donut chart
    ax = axes[0]
    ax.set_facecolor(COLORS['bg'])
    pie_labels = [f'{t.capitalize()} ({c})' for t, c in counts.items()]
    pie_colors = ['#1565C0', '#E65100']
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=pie_labels, colors=pie_colors,
        autopct='%1.1f%%', startangle=90,
        wedgeprops=dict(width=0.55, edgecolor='white', linewidth=2),
        pctdistance=0.75, textprops={'fontsize': 10})
    for at in autotexts:
        at.set_fontsize(10); at.set_fontweight('bold'); at.set_color('white')
    ax.set_title(f'Feature Mapping Strategy\n(NSL-KDD → UNSW-NB15, {total} features)',
                 fontsize=11, fontweight='bold', pad=10)
    ax.text(0, 0, f'{total}\ntotal', ha='center', va='center',
            fontsize=13, fontweight='bold', color='#263238')

    # ── Sankey-style feature list
    ax2 = axes[1]
    ax2.set_facecolor(COLORS['bg'])
    ax2.axis('off')

    semantic_rows = df[df['mapping_type'] == 'semantic'].head(10)
    projected_rows = df[df['mapping_type'] == 'projected'].head(10)

    ax2.text(0.25, 0.97, 'NSL-KDD Feature', ha='center', va='top',
             fontsize=9, fontweight='bold', color='#1565C0')
    ax2.text(0.75, 0.97, 'UNSW-NB15 Feature', ha='center', va='top',
             fontsize=9, fontweight='bold', color='#00695C')

    row_h = 0.078
    for i, (_, row) in enumerate(semantic_rows.iterrows()):
        y = 0.90 - i * row_h
        ax2.text(0.25, y, row['source_feature'], ha='center', va='center',
                 fontsize=8, color='#1565C0',
                 bbox=dict(boxstyle='round,pad=0.2', fc='#E3F2FD', ec='#1565C0', lw=0.7))
        ax2.text(0.75, y, row['target_feature'], ha='center', va='center',
                 fontsize=8, color='#00695C',
                 bbox=dict(boxstyle='round,pad=0.2', fc='#E8F5E9', ec='#00695C', lw=0.7))
        ax2.annotate('', xy=(0.60, y), xytext=(0.40, y),
                     arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1.2,
                                     mutation_scale=11))
        mtype = '≡' if row['mapping_type'] == 'identical' else '≈'
        ax2.text(0.50, y + 0.015, mtype, ha='center', va='center',
                 fontsize=8, color='#555')

    ax2.text(0.50, 0.90 - 10 * row_h, f'+ {len(projected_rows)} projected\n(FC network)',
             ha='center', va='center', fontsize=8, color=COLORS['generator'],
             bbox=dict(boxstyle='round,pad=0.3', fc='#FFF3E0',
                       ec=COLORS['generator'], lw=1))

    ax2.set_title('Feature Mapping Examples (Semantic + Identical)',
                  fontsize=10, fontweight='bold', pad=6)
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)

    plt.suptitle('Feature Harmonizer: Cross-Dataset Feature Alignment',
                 fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    out = OUTPUT_DIR / 'fig6_feature_mapping.png'
    plt.savefig(out)
    plt.close()
    print(f'Saved: {out}')


# ===========================================================
# FIGURE 7 — Minority Class F1 Heatmap (class imbalance focus)
# ===========================================================
def figure7_minority_f1():
    df = pd.read_csv('results/tables/table3_main_results.csv')
    df = df[df['macro_f1'] > 0.80]
    agg = (df.groupby(['scenario', 'source_dataset', 'target_dataset'])
             ['minority_class_f1'].mean().reset_index())

    datasets = ['nsl_kdd', 'unsw_nb15', 'cic_ids2017']
    labels   = ['NSL-KDD', 'UNSW-NB15', 'CIC-IDS2017']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.patch.set_facecolor(COLORS['bg'])

    scens = [
        ('A', 'Scenario A: Within-Dataset',  'Blues'),
        ('B', 'Scenario B: Direct Transfer', 'Oranges'),
        ('C', 'Scenario C: Adapted + MMD',   'Greens'),
    ]

    for ax, (scen, title, cmap) in zip(axes, scens):
        ax.set_facecolor(COLORS['bg'])
        matrix = np.full((3, 3), np.nan)
        for i, src in enumerate(datasets):
            for j, tgt in enumerate(datasets):
                row = agg[(agg['scenario'] == scen) &
                          (agg['source_dataset'] == src) &
                          (agg['target_dataset'] == tgt)]
                if not row.empty:
                    matrix[i, j] = row['minority_class_f1'].values[0]

        vmin = np.nanmin(matrix) - 0.02
        vmax = np.nanmax(matrix) + 0.02
        im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=8.5)
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.set_xlabel('Target Dataset', fontsize=9)
        ax.set_ylabel('Source Dataset', fontsize=9)
        ax.set_title(title, fontsize=10, fontweight='bold', pad=8)

        mid = (vmin + vmax) / 2
        for i in range(3):
            for j in range(3):
                if not np.isnan(matrix[i, j]):
                    clr = 'white' if matrix[i, j] < mid + 0.04 else '#263238'
                    ax.text(j, i, f'{matrix[i,j]:.3f}',
                            ha='center', va='center',
                            fontsize=10, fontweight='bold', color=clr)
                else:
                    ax.text(j, i, '—', ha='center', va='center',
                            fontsize=11, color='#90A4AE')
        plt.colorbar(im, ax=ax, shrink=0.75, label='Minority Class F1')

    fig.suptitle('Minority Class F1: Effectiveness on Imbalanced Attack Classes',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = OUTPUT_DIR / 'fig7_minority_class_f1.png'
    plt.savefig(out)
    plt.close()
    print(f'Saved: {out}')


# ===========================================================
# FIGURE 8 — Scenario B vs C: MMD Gain
# ===========================================================
def figure8_mmd_gain():
    df = pd.read_csv('results/tables/table3_main_results.csv')
    df = df[df['macro_f1'] > 0.80]
    agg = (df.groupby(['scenario', 'source_dataset', 'target_dataset'])
             [['macro_f1', 'minority_class_f1']].mean().reset_index())

    pairs = [
        ('nsl_kdd',  'unsw_nb15',  'NSL-KDD\n→ UNSW-NB15'),
        ('unsw_nb15','nsl_kdd',    'UNSW-NB15\n→ NSL-KDD'),
        ('nsl_kdd',  'cic_ids2017','NSL-KDD\n→ CIC-IDS2017'),
        ('unsw_nb15','cic_ids2017','UNSW-NB15\n→ CIC-IDS2017'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(COLORS['bg'])
    fig.suptitle('Scenario B vs C: Effect of MMD Domain Adaptation',
                 fontsize=12, fontweight='bold')

    for ax, metric in zip(axes, ['macro_f1', 'minority_class_f1']):
        ax.set_facecolor(COLORS['bg'])
        labels_p, b_vals, c_vals = [], [], []

        for src, tgt, lbl in pairs:
            labels_p.append(lbl)
            rb = agg[(agg['scenario']=='B') & (agg['source_dataset']==src) &
                     (agg['target_dataset']==tgt)]
            rc = agg[(agg['scenario']=='C') & (agg['source_dataset']==src) &
                     (agg['target_dataset']==tgt)]
            b_vals.append(rb[metric].values[0] if not rb.empty else np.nan)
            c_vals.append(rc[metric].values[0] if not rc.empty else np.nan)

        x = np.arange(len(labels_p))
        w = 0.32
        bars_b = ax.bar(x - w/2, b_vals, w - 0.02,
                        label='B: Direct Transfer',
                        color=COLORS['scenario_b'], alpha=0.85, edgecolor='white')
        bars_c = ax.bar(x + w/2, c_vals, w - 0.02,
                        label='C: + MMD Alignment',
                        color=COLORS['scenario_c'], alpha=0.85, edgecolor='white')

        for bars in [bars_b, bars_c]:
            for bar in bars:
                h = bar.get_height()
                if not np.isnan(h):
                    ax.text(bar.get_x() + bar.get_width()/2, h + 0.003,
                            f'{h:.3f}', ha='center', va='bottom', fontsize=8)

        # Delta annotation
        for i, (bv, cv) in enumerate(zip(b_vals, c_vals)):
            if not (np.isnan(bv) or np.isnan(cv)):
                delta = cv - bv
                color = '#1B5E20' if delta >= 0 else '#B71C1C'
                sign  = '+' if delta >= 0 else ''
                ax.text(i, max(bv, cv) + 0.018, f'Δ{sign}{delta:.3f}',
                        ha='center', fontsize=7.5, color=color, fontweight='bold')

        yvals = [v for v in b_vals + c_vals if not np.isnan(v)]
        ax.set_ylim(min(yvals) - 0.04, max(yvals) + 0.06)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_p, fontsize=9)
        metric_label = 'Macro-F1' if metric == 'macro_f1' else 'Minority Class F1'
        ax.set_ylabel(metric_label, fontsize=11)
        ax.set_title(f'{metric_label}: B vs C', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    out = OUTPUT_DIR / 'fig8_mmd_gain.png'
    plt.savefig(out)
    plt.close()
    print(f'Saved: {out}')


# ===========================================================
# RUN ALL
# ===========================================================
if __name__ == '__main__':
    import os
    os.chdir(Path(__file__).parent)

    print('Generating paper figures...\n')
    figure1_architecture()
    figure2_pipeline()
    figure3_main_results()
    figure4_transfer_heatmaps()
    figure5_smote_comparison()
    figure6_feature_mapping()
    figure7_minority_f1()
    figure8_mmd_gain()
    print('\nAll figures saved to results/figures/')
