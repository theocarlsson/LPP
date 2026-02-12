import subprocess
import re
import matplotlib.pyplot as plt

# Configuration
executable = "/LPP/skeleton/Assignment2/LPP/demo/demo"  # Replace with your compiled executable
scenario_file = "hugeScenario.xml"
max_steps = 500
implementations = ["seq", "simd", "omp"]
thread_counts = [1, 2, 4, 8, 12]  # Only relevant for omp/simd

# Function to run the simulation and parse time
def run_simulation(impl, threads):
    cmd = [executable, "--timing-mode", f"--max-steps={max_steps}"]
    
    if impl == "seq":
        cmd.append("--seq")
    elif impl == "simd":
        cmd.append("--simd")
        cmd += ["--max-threads", str(threads)]
    elif impl == "omp":
        cmd.append("--omp")
        cmd += ["--max-threads", str(threads)]
    
    cmd.append(scenario_file)
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Try to extract "Target average time: XXX ms" from output
    match = re.search(r"Target average time: ([0-9.]+) ms", result.stdout)
    if match:
        return float(match.group(1))
    else:
        print("Warning: Could not parse output")
        print(result.stdout)
        return None

# Store results
results = {impl: [] for impl in implementations}

# Run simulations
for impl in implementations:
    if impl == "seq":
        # Sequential does not depend on threads
        time_ms = run_simulation(impl, 1)
        results[impl] = [time_ms] * len(thread_counts)
    else:
        for threads in thread_counts:
            time_ms = run_simulation(impl, threads)
            results[impl].append(time_ms)

# Plot results
plt.figure(figsize=(10,6))
for impl in implementations:
    plt.plot(thread_counts, results[impl], marker='o', label=impl.upper())

plt.xlabel("Number of Threads")
plt.ylabel("Execution Time (ms)")
plt.title("Crowd Simulation Performance")
plt.xticks(thread_counts)
plt.grid(True)
plt.legend()
plt.savefig("crowd_sim_performance.png", dpi=300)
print("Plot saved as crowd_sim_performance.png")
