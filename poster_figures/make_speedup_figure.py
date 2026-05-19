import matplotlib.pyplot as plt

# ---- Data ----
ctx_all = [8192, 16384, 32768, 65536, 131072]

# Attention-only speedup
attn_single = [1.74952381, 2.512972572, 3.605633803, 4.746748279, 5.64567474]
attn_double_x = [8192, 16384, 32768, 65536]
attn_double   = [1.993932039, 2.979020979, 4.128795462, 5.270887166]

# End-to-end speedup
e2e_single = [1.03, 1.07, 1.17, 1.35, 1.70]
e2e_double_x = [8192, 16384, 32768, 65536]
e2e_double   = [1.07, 1.16, 1.34, 1.68]

COLOR_B1 = '#1f4e79'
COLOR_B2 = '#c45a00'

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10.4, 3.8))


def style_axis(ax, title, ylim):
    ax.set_xscale('log', base=2)
    ax.set_xticks(ctx_all)
    ax.set_xticklabels(['8K', '16K', '32K', '64K', '128K'])
    ax.set_xlabel('Context length', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3, linewidth=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=1.2)


# ---- Left: Attention speedup ----
ax_l.plot(ctx_all, attn_single, marker='o', markersize=8, linewidth=2.5,
          label='Batch 1', color=COLOR_B1)
ax_l.plot(attn_double_x, attn_double, marker='s', markersize=8, linewidth=2.5,
          label='Batch 2', color=COLOR_B2)
ax_l.scatter([131072], [6.2], marker='x', s=170, color=COLOR_B2, linewidths=2.8)
ax_l.annotate('Full-KV OOM', xy=(131072, 6.2), xytext=(0, 10),
              textcoords='offset points', ha='center', va='bottom',
              fontsize=9.5, color=COLOR_B2, fontweight='bold')
ax_l.annotate(f'{attn_single[-1]:.2f}x', xy=(131072, attn_single[-1]),
              xytext=(-4, -18), textcoords='offset points',
              ha='center', fontsize=12, fontweight='bold', color=COLOR_B1)
ax_l.annotate(f'{attn_double[-1]:.2f}x', xy=(65536, attn_double[-1]),
              xytext=(0, 12), textcoords='offset points',
              ha='center', fontsize=12, fontweight='bold', color=COLOR_B2)
ax_l.text(8192, 1.18, 'Full-KV (1x)', fontsize=9, color='gray')
style_axis(ax_l, 'Attention speedup', (0.5, 7.0))
ax_l.set_ylabel('Speedup over Full-KV  (x)', fontsize=11)
ax_l.legend(loc='upper left', frameon=False, fontsize=10)


# ---- Right: End-to-end speedup ----
ax_r.plot(ctx_all, e2e_single, marker='o', markersize=8, linewidth=2.5,
          label='Batch 1', color=COLOR_B1)
ax_r.plot(e2e_double_x, e2e_double, marker='s', markersize=8, linewidth=2.5,
          label='Batch 2', color=COLOR_B2)
ax_r.scatter([131072], [1.85], marker='x', s=170, color=COLOR_B2, linewidths=2.8)
ax_r.annotate('Full-KV OOM', xy=(131072, 1.85), xytext=(0, 10),
              textcoords='offset points', ha='center', va='bottom',
              fontsize=9.5, color=COLOR_B2, fontweight='bold')
ax_r.annotate(f'{e2e_single[-1]:.2f}x', xy=(131072, e2e_single[-1]),
              xytext=(-4, 12), textcoords='offset points',
              ha='center', fontsize=12, fontweight='bold', color=COLOR_B1)
ax_r.annotate(f'{e2e_double[-1]:.2f}x', xy=(65536, e2e_double[-1]),
              xytext=(0, 12), textcoords='offset points',
              ha='center', fontsize=12, fontweight='bold', color=COLOR_B2)
ax_r.text(8192, 0.92, 'Full-KV (1x)', fontsize=9, color='gray')
style_axis(ax_r, 'End-to-end decode speedup', (0.85, 2.15))
ax_r.set_ylabel('Speedup over Full-KV  (x)', fontsize=11)
ax_r.legend(loc='upper left', frameon=False, fontsize=10)


plt.tight_layout()
plt.savefig('/home/yoongonkim/DCT-Page/poster_figures/dct_speedup.pdf',
            bbox_inches='tight')
plt.savefig('/home/yoongonkim/DCT-Page/poster_figures/dct_speedup.png',
            dpi=300, bbox_inches='tight')
print("Saved: dct_speedup.pdf, dct_speedup.png")
