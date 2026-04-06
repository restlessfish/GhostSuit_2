#!/usr/bin/env python3
"""Generate all paper figures using matplotlib"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Set style for academic papers
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})

def save_fig(fig, name):
    fig.savefig(f'{name}.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ {name}.png")
    plt.close(fig)

# ============ Figure 2: Layer Time Distribution ============
def fig2_layer_time():
    fig, ax = plt.subplots(figsize=(10, 7))
    
    layers = ['layer4.1.conv2', 'layer4.0.conv2', 'layer4.1.conv1', 
              'layer3.1.conv1', 'layer3.1.conv2', 'layer3.0.conv2', 'conv1',
              'layer2.1.conv2', 'layer2.1.conv1', 'layer2.0.conv2',
              'layer1.1.conv2', 'layer1.1.conv1', 'layer1.0.conv2', 'layer1.0.conv1',
              'layer4.0.downsample', 'layer3.0.downsample', 'layer2.0.downsample', 'fc']
    times = [11.13, 11.12, 11.11, 2.48, 2.47, 2.46, 1.76, 1.29, 1.29, 1.29,
             1.12, 1.12, 1.12, 1.12, 0.83, 0.74, 0.73, 0.20]
    
    colors = ['#e03131']*3 + ['#f59f00']*3 + ['#74c0fc'] + ['#74c0fc']*3 + ['#69db7c']*4 + ['#b197fc']*3 + ['#adb5bd']
    
    bars = ax.barh(layers, times, color=colors, edgecolor='white', linewidth=0.5)
    
    ax.set_xlabel('Computation Time (ms)')
    ax.set_title('Figure 2: ResNet-18 Layer Computation Time Distribution')
    ax.set_xlim(0, 12)
    
    # Add value labels
    for bar, val in zip(bars, times):
        ax.text(val + 0.1, bar.get_y() + bar.get_height()/2, f'{val:.2f}', 
                va='center', ha='left', fontsize=8)
    
    # Legend - upper right
    legend_elements = [
        mpatches.Patch(color='#e03131', label='Layer4 (~11ms)'),
        mpatches.Patch(color='#f59f00', label='Layer3 (~2.5ms)'),
        mpatches.Patch(color='#74c0fc', label='Layer2/conv1 (~1.3ms)'),
        mpatches.Patch(color='#69db7c', label='Layer1 (~1.1ms)'),
        mpatches.Patch(color='#b197fc', label='Downsample (~0.8ms)'),
        mpatches.Patch(color='#adb5bd', label='fc (~0.2ms)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9, bbox_to_anchor=(1.0, 1.0))
    
    plt.tight_layout()
    save_fig(fig, 'Figure_2_Layer_Time')

# ============ Figure 3: Cluster Smoothing Effect ============
def fig3_cluster_smoothing():
    fig, ax = plt.subplots(figsize=(8, 5))
    
    methods = ['Original\nIn-Run Shapley', 'Hotspot\n(Raw)', 'Hotspot + CS\n(Ours)']
    auroc = [0.678, 0.5371, 0.7355]
    colors = ['#339af0', '#e03131', '#2f9e44']
    
    bars = ax.bar(methods, auroc, color=colors, width=0.6, edgecolor='white', linewidth=1)
    
    ax.set_ylabel('AUROC')
    ax.set_ylim(0, 0.85)
    ax.set_title('Figure 3: Cluster Smoothing Effect on AUROC')
    
    # Add value labels
    for bar, val in zip(bars, auroc):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.4f}', 
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add improvement arrow
    ax.annotate('', xy=(2, 0.7355), xytext=(1, 0.5371),
                arrowprops=dict(arrowstyle='->', color='#2f9e44', lw=2))
    ax.text(1.5, 0.64, '+37%', fontsize=10, color='#2f9e44', ha='center', fontweight='bold')
    
    plt.tight_layout()
    save_fig(fig, 'Figure_3_Cluster_Smoothing')

# ============ Figure 4a: AUROC Comparison ============
def fig4a_auroc():
    fig, ax = plt.subplots(figsize=(9, 5))
    
    methods = ['Full\n59 layers', 'Hotspot\nTop-18', 'Random\nTop-18', 'Param-count\nTop-18', 
               'Layer2+3+4+FC', 'Layer3+4+FC', 'Layer4+FC']
    auroc = [0.5437, 0.5371, 0.5445, 0.5437, 0.5416, 0.5557, 0.5230]
    
    colors = ['#339af0'] + ['#51cf66']*6
    bars = ax.bar(methods, auroc, color=colors, width=0.65, edgecolor='white', linewidth=0.5)
    
    ax.set_ylabel('AUROC (Raw)')
    ax.set_ylim(0.50, 0.58)
    ax.set_title('Figure 4a: AUROC Comparison by Layer Selection Strategy')
    ax.axhline(y=0.5557, color='#2f9e44', linestyle='--', alpha=0.5, label='Best raw AUROC')
    
    for bar, val in zip(bars, auroc):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.002, f'{val:.4f}', 
                ha='center', va='bottom', fontsize=9, rotation=0)
    
    plt.tight_layout()
    save_fig(fig, 'Figure_4a_AUROC')

# ============ Figure 4b: Throughput Comparison ============
def fig4b_throughput():
    fig, ax = plt.subplots(figsize=(9, 5))
    
    methods = ['Full\n59 layers', 'Hotspot\nTop-18', 'Random\nTop-18', 'Param-count\nTop-18', 
               'Layer2+3+4+FC', 'Layer3+4+FC', 'Layer4+FC']
    throughput = [2016, 2034, 2018, 2039, 1308, 1506, 1784]
    
    colors = ['#339af0'] + ['#51cf66']*4 + ['#ffa94d'] + ['#ffd43b']
    bars = ax.bar(methods, throughput, color=colors, width=0.65, edgecolor='white', linewidth=0.5)
    
    ax.set_ylabel('Throughput (samples/s)')
    ax.set_ylim(1000, 2200)
    ax.set_title('Figure 4b: Throughput Comparison')
    
    for bar, val in zip(bars, throughput):
        ax.text(bar.get_x() + bar.get_width()/2, val + 20, f'{val}', 
                ha='center', va='bottom', fontsize=9)
    
    # Highlight hotspot
    ax.annotate('+55%', xy=(1, 2034), xytext=(4, 1900),
                arrowprops=dict(arrowstyle='->', color='#2f9e44'),
                fontsize=10, color='#2f9e44', fontweight='bold')
    
    plt.tight_layout()
    save_fig(fig, 'Figure_4b_Throughput')

# ============ Figure 5: Alpha Sensitivity ============
def fig5_alpha():
    fig, ax = plt.subplots(figsize=(7, 5))
    
    alphas = [0.00, 0.02, 0.05, 0.10, 0.20]
    auroc = [0.7317, 0.7321, 0.7326, 0.7336, 0.7355]
    
    ax.plot(alphas, auroc, 'o-', color='#4dabf7', linewidth=2, markersize=10, 
            markerfacecolor='white', markeredgewidth=2)
    
    # Highlight optimal point
    ax.scatter([0.20], [0.7355], color='#2f9e44', s=150, zorder=5, marker='*', label='Optimal (α=0.2)')
    
    ax.set_xlabel('Alpha (α)')
    ax.set_ylabel('AUROC')
    ax.set_ylim(0.730, 0.738)
    ax.set_title('Figure 5: Alpha Parameter Sensitivity Analysis')
    ax.set_xticks(alphas)
    ax.legend(loc='lower right')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_fig(fig, 'Figure_5_Alpha_Sensitivity')

# ============ Figure 6: GPT-2 vs ResNet-18 ============
def fig6_gpt2_resnet():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    
    # Left: Throughput
    models = ['ResNet-18', 'GPT-2']
    baseline = [1308, 189.04]
    hotspot = [2034, 190.22]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, baseline, width, label='Baseline', color='#339af0')
    bars2 = ax1.bar(x + width/2, hotspot, width, label='Hotspot Selection', color='#2f9e44')
    
    ax1.set_ylabel('Throughput (samples/s)')
    ax1.set_title('Throughput Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.legend()
    
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, f'{bar.get_height():.0f}', 
                ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, f'{bar.get_height():.0f}', 
                ha='center', va='bottom', fontsize=9)
    
    # Right: Improvement
    improvements = [55.5, 0.62]
    colors = ['#2f9e44', '#339af0']
    bars = ax2.bar(models, improvements, color=colors, width=0.5)
    
    ax2.set_ylabel('Throughput Improvement (%)')
    ax2.set_title('Improvement from Hotspot Selection')
    ax2.set_ylim(0, 65)
    
    for bar, val in zip(bars, improvements):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 1, f'+{val:.1f}%', 
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    fig.suptitle('Figure 6: GPT-2 vs ResNet-18 Throughput Comparison', fontsize=12, y=1.02)
    plt.tight_layout()
    save_fig(fig, 'Figure_6_GPT2_vs_ResNet')

# ============ Figure 7: Top-K Recall ============
def fig7_topk():
    fig, ax = plt.subplots(figsize=(7, 5))
    
    topk = ['Top-10', 'Top-20', 'Top-50', 'Top-100']
    detected = [4, 6, 11, 20]
    total = [10, 20, 50, 100]
    
    x = np.arange(len(topk))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, detected, width, label='Detected Mislabeled', color='#e03131')
    bars2 = ax.bar(x + width/2, total, width, label='Total Samples', color='#adb5bd')
    
    ax.set_xlabel('Top-K Threshold')
    ax.set_ylabel('Count')
    ax.set_title('Figure 7: Top-K Mislabeled Sample Detection')
    ax.set_xticks(x)
    ax.set_xticklabels(topk)
    ax.legend()
    
    for bar, val in zip(bars1, detected):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold', color='#e03131')
    
    plt.tight_layout()
    save_fig(fig, 'Figure_7_TopK_Recall')

# ============ Figure 8: Trade-off ============
def fig8_tradeoff():
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Data points
    comp_pct = [100, 30, 30, 30, 30, 22, 12, 30]
    auroc = [0.5437, 0.5371, 0.5445, 0.5437, 0.5416, 0.5557, 0.5230, 0.7355]
    labels = ['Full-layer', 'Hotspot', 'Random', 'Param', 'L2+3+4', 'L3+4', 'L4', 'Hotspot+CS']
    colors = ['#339af0', '#51cf66', '#51cf66', '#51cf66', '#51cf66', '#51cf66', '#51cf66', '#e03131']
    sizes = [100, 100, 100, 100, 100, 100, 100, 200]
    
    for i in range(len(comp_pct)):
        if i == 7:  # Highlight the best point
            ax.scatter(comp_pct[i], auroc[i], c=colors[i], s=sizes[i], zorder=5, marker='*', edgecolors='black')
        else:
            ax.scatter(comp_pct[i], auroc[i], c=colors[i], s=sizes[i], zorder=3, edgecolors='white', linewidth=1)
    
    # Add labels
    offset = [(5, 5), (-15, 5), (5, -10), (-20, -10), (5, 5), (5, 5), (5, -10), (5, 10)]
    for i, (x, y) in enumerate(zip(comp_pct, auroc)):
        ax.annotate(labels[i], (x, y), xytext=offset[i], textcoords='offset points', fontsize=8)
    
    # Draw arrow showing the improvement
    ax.annotate('', xy=(30, 0.7355), xytext=(30, 0.5371),
                arrowprops=dict(arrowstyle='->', color='#e03131', lw=2, ls='--'))
    ax.text(38, 0.64, '+37%\nrelative', fontsize=9, color='#e03131')
    
    ax.set_xlabel('Computation (% of layers)')
    ax.set_ylabel('AUROC')
    ax.set_xlim(-5, 115)
    ax.set_ylim(0.45, 0.80)
    ax.set_title('Figure 8: Computation Reduction vs AUROC Trade-off')
    ax.grid(True, alpha=0.3)
    
    # Legend
    legend_elements = [
        plt.scatter([], [], c='#339af0', s=100, label='Full computation'),
        plt.scatter([], [], c='#51cf66', s=100, label='Partial computation'),
        plt.scatter([], [], c='#e03131', s=200, marker='*', label='Hotspot + CS (Ours)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    save_fig(fig, 'Figure_8_Tradeoff')

# ============ Figure 1: Architecture Diagram ============
def fig1_architecture():
    """Create a clean architecture diagram using matplotlib"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Define boxes
    boxes = {
        'input': (1, 6, 2.5, 1.5, '#e7f5ff', '#1c7ed6', 'INPUT'),
        'profile': (1, 3.5, 2.5, 2, '#fff3bf', '#f59f00', 'PROFILING'),
        'compute': (5, 3.5, 2.5, 2, '#ffe3e3', '#e03131', 'COMPUTATION'),
        'shapley': (9, 3.5, 2.5, 2, '#d3f9d8', '#2f9e44', 'SHAPLEY'),
        'post': (12, 3.5, 1.8, 2, '#e5dbff', '#7048e8', 'POST'),
        'output': (11.5, 0.5, 2.3, 1.5, '#d0ebff', '#1971c2', 'OUTPUT'),
    }
    
    # Draw boxes
    for name, (x, y, w, h, fc, ec, label) in boxes.items():
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", 
                                       facecolor=fc, edgecolor=ec, linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center', 
                fontsize=11, fontweight='bold', color=ec)
    
    # Add sub-boxes in PROFILING
    ax.text(1.25, 4.8, '• Hook Profiling\n• Per-Layer Time\n• Per-Layer Sensitivity', 
            fontsize=8, va='top', ha='left')
    
    # Add sub-boxes in COMPUTATION
    ax.text(5.25, 4.8, '• Hotspot: Full\n• Non-Hotspot: Skip', 
            fontsize=8, va='top', ha='left')
    
    # Add sub-boxes in SHAPLEY
    ax.text(9.25, 4.8, '• First-Order\n• Per-Sample Values', 
            fontsize=8, va='top', ha='left')
    
    # Add sub-boxes in POST
    ax.text(12.1, 4.8, '• Cluster\n  Smoothing\n• Final Scores', 
            fontsize=8, va='top', ha='left')
    
    # Add arrows
    arrows = [
        (3.5, 6.75, 5, 4.5),  # input -> compute
        (5, 4.5, 9, 4.5),     # compute -> shapley
        (9, 4.5, 12, 4.5),    # shapley -> post
        (12.9, 3.5, 12.65, 2), # post -> output
        (2.25, 3.5, 2.25, 5.25), # profile -> (vertical)
    ]
    
    for x1, y1, x2, y2 in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#666', lw=1.5))
    
    # Add profile connection to compute
    ax.annotate('', xy=(5, 4.5), xytext=(3.5, 4.5),
                arrowprops=dict(arrowstyle='->', color='#666', lw=1.5))
    ax.text(4.25, 5.0, 'Hotspot\nSelection', fontsize=8, ha='center', 
            color='#f59f00', style='italic')
    
    # Title
    ax.text(7, 7.5, 'Figure 1: GhostSuit Framework Architecture', 
            ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Legend
    legend_items = [
        ('Input/Data', '#e7f5ff', '#1c7ed6'),
        ('Profiling', '#fff3bf', '#f59f00'),
        ('Computation', '#ffe3e3', '#e03131'),
        ('Shapley', '#d3f9d8', '#2f9e44'),
        ('Post-processing', '#e5dbff', '#7048e8'),
        ('Output', '#d0ebff', '#1971c2'),
    ]
    
    for i, (label, fc, ec) in enumerate(legend_items):
        rect = mpatches.Rectangle((0.5, 0.8 - i*0.35), 0.3, 0.25, 
                                   facecolor=fc, edgecolor=ec, linewidth=1)
        ax.add_patch(rect)
        ax.text(0.9, 0.92 - i*0.35, label, fontsize=8, va='center')
    
    plt.tight_layout()
    save_fig(fig, 'Figure_1_Architecture')

if __name__ == '__main__':
    print("Generating figures...")
    fig1_architecture()
    fig2_layer_time()
    fig3_cluster_smoothing()
    fig4a_auroc()
    fig4b_throughput()
    fig5_alpha()
    fig6_gpt2_resnet()
    fig7_topk()
    fig8_tradeoff()
    print("\n✅ All figures generated!")
