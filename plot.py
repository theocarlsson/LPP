#!/usr/bin/env python3
"""
Simple speedup plot: SEQ vs REGION
Run from the LPP root: python3 speedup.py
"""

import subprocess, re, sys
import matplotlib.pyplot as plt
import numpy as np

DEMO     = "./demo/demo"
SCENARIO = "scenario.xml"
STEPS    = 1000
REPEATS  = 3
THREADS  = 8

def run(flag):
    cmd = [DEMO, "--timing-mode", flag,
           f"--max-steps={STEPS}", f"--max-threads={THREADS}", SCENARIO]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    txt = out.stdout + out.stderr
    seq  = re.search(r"SEQ average time:\s+([\d.]+)", txt)
    tgt  = re.search(r"Target average time:\s+([\d.]+)", txt)
    spd  = re.search(r"Speedup:\s+([\d.]+)", txt)
    return (float(seq.group(1)) if seq else None,
            float(tgt.group(1)) if tgt else None,
            float(spd.group(1)) if spd else None)

print(f"Running {REPEATS} repeats ({STEPS} steps each)...\n")

seq_times, reg_times, speedups = [], [], []
for i in range(REPEATS):
    s, r, sp = run("--region")
    if s and r and sp:
        seq_times.append(s); reg_times.append(r); speedups.append(sp)
        print(f"  Run {i+1}: SEQ={s:.0f}ms  REGION={r:.0f}ms  Speedup={sp:.2f}x")

if not speedups:
    print("No results — check demo path and scenario."); sys.exit(1)

avg_seq = np.mean(seq_times)
avg_reg = np.mean(reg_times)
avg_spd = np.mean(speedups)

print(f"\nAverage SEQ:    {avg_seq:.0f} ms")
print(f"Average REGION: {avg_reg:.0f} ms")
print(f"Average Speedup: {avg_spd:.2f}x")

# ── plot ──────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle(f"SEQ vs REGION  ({STEPS} steps, {THREADS} threads)", fontsize=13)

# Bar: avg time
ax1.bar(["SEQ", "REGION"], [avg_seq, avg_reg],
        color=["#4a90d9", "#e8524a"], width=0.5)
ax1.set_ylabel("Time (ms)")
ax1.set_title("Average Execution Time")
for x, v in enumerate([avg_seq, avg_reg]):
    ax1.text(x, v + avg_seq*0.01, f"{v:.0f} ms", ha="center", fontweight="bold")

# Bar: speedup per run + average line
runs = list(range(1, REPEATS+1))
ax2.bar(runs, speedups, color="#50c878", width=0.5)
ax2.axhline(avg_spd, color="orange", linewidth=2, linestyle="--",
            label=f"avg {avg_spd:.2f}x")
ax2.axhline(1.0, color="gray", linewidth=1, linestyle=":")
ax2.set_xlabel("Run")
ax2.set_ylabel("Speedup (×)")
ax2.set_title("Speedup per Run")
ax2.legend()
ax2.set_xticks(runs)

plt.tight_layout()
plt.savefig("speedup.png", dpi=150, bbox_inches="tight")
print("\nSaved: speedup.png")
plt.show()