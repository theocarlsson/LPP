import struct
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/LPP/skeleton/Assignment2/LPP/visualizer')
from visualize_export import deserialize, ReadSingleFrame

filename = '/LPP/skeleton/Assignment2/LPP/demo/export_trace.bin'
output_dir = '/LPP/skeleton/Assignment2/LPP/demo/frames'
os.makedirs(output_dir, exist_ok=True)

with open(filename, 'rb') as file:
    frames, heatmaps, offsets = deserialize(file, 1000, True)

    for step in range(len(offsets)):
        agents, heatmap = ReadSingleFrame(file, offsets, step)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_xlim(0, 160)
        ax.set_ylim(0, 120)
        ax.invert_yaxis()
        ax.set_aspect('equal')

        if agents:
            xs, ys = zip(*agents)
            ax.scatter(xs, ys, s=10, color='green', edgecolor='black')

        ax.set_title(f'Step {step+1}/{len(offsets)}')
        fig.savefig(f'{output_dir}/frame_{step:04d}.png', dpi=100, bbox_inches='tight')
        plt.close(fig)

        if step % 20 == 0:
            print(f'Saved frame {step+1}/{len(offsets)}')

print("Done! Converting to video...")
os.system(f'ffmpeg -r 10 -i {output_dir}/frame_%04d.png -vcodec libx264 -pix_fmt yuv420p output.mp4')
print("Video saved as output.mp4")
