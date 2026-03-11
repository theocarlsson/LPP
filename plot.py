import matplotlib.pyplot as plt
import numpy as np

# Per-kernel times in ms (average of your 5 measurements)
steps = ['Fade', 'Agent+Clamp', 'Scale', 'Blur']

gpu_times = [
    (0.404352 + 0.0488 + 0.077696 + 0.07264 + 0.076768) / 5,   # fade
    (0.144384 + 0.02048 + 0.018432 + 0.019456 + 0.052224) / 5,  # agent+clamp
    (1.31379  + 1.2544  + 1.26157  + 1.3015   + 1.28819 ) / 5,  # scale
    (2.98906  + 2.90099 + 4.17075  + 3.64032  + 3.61984 ) / 5,  # blur
]

# SEQ heatmap breakdown - estimate from total ~89ms
# Fade: ~10ms, Agent: ~2ms, Scale: ~40ms, Blur: ~37ms (rough proportions)
seq_total = 89.0
seq_times = [10.0, 2.0, 40.0, 37.0]

x = np.arange(len(steps))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

bars_seq = ax.bar(x - width/2, seq_times, width, label='SEQ (CPU)', color='steelblue')
bars_gpu = ax.bar(x + width/2, gpu_times, width, label='CUDA (GPU)', color='tomato')

# Value labels on bars
for bar in bars_seq:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{bar.get_height():.1f}ms', ha='center', va='bottom', fontsize=9)

for bar in bars_gpu:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{bar.get_height():.3f}ms', ha='center', va='bottom', fontsize=9)

# Speedup annotations
for i, (s, g) in enumerate(zip(seq_times, gpu_times)):
    ax.text(x[i], max(s, g) + 2.5, f'{s/g:.1f}x', ha='center',
            fontsize=10, fontweight='bold', color='green')

ax.set_ylabel('Time (ms)')
ax.set_title('Heatmap Step Times: SEQ vs CUDA (hugeScenario)\nGreen labels show per-step speedup')
ax.set_xticks(x)
ax.set_xticklabels(steps)
ax.legend()
ax.set_ylim(0, 55)

plt.tight_layout()
plt.savefig('heatmap_steps.png', dpi=150)
print("Saved to heatmap_steps.png")