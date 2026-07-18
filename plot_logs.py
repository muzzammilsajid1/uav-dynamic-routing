import re
import matplotlib.pyplot as plt

log_file_path = r"C:\Users\Fast Computer\.gemini\antigravity\brain\424d0f7c-a878-4797-a298-31a0e845f972\.system_generated\tasks\task-168.log"

timesteps = []
ep_rew_means = []

with open(log_file_path, "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

current_timestep = None
current_rew_mean = None

# We look for blocks like:
# | rollout/            |          |
# |    ep_rew_mean      | -130     |
# ...
# | time/               |          |
# ...
# |    total_timesteps  | 52       |

for line in lines:
    if "ep_rew_mean" in line:
        try:
            val = float(line.split("|")[2].strip())
            current_rew_mean = val
        except:
            pass
    if "total_timesteps" in line:
        try:
            val = int(line.split("|")[2].strip())
            current_timestep = val
        except:
            pass
            
    # When we hit the end of a block, if we have both, we save them.
    # A block ends with "----------------------------------"
    if line.startswith("----------------------------------"):
        if current_timestep is not None and current_rew_mean is not None:
            timesteps.append(current_timestep)
            ep_rew_means.append(current_rew_mean)
        current_timestep = None
        current_rew_mean = None

print(f"Extracted {len(timesteps)} data points.")

if len(timesteps) > 0:
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, ep_rew_means, label='ep_rew_mean', color='blue')
    
    # Smooth curve
    if len(timesteps) > 20:
        import numpy as np
        window = 20
        smoothed = np.convolve(ep_rew_means, np.ones(window)/window, mode='valid')
        plt.plot(timesteps[window-1:], smoothed, label='ep_rew_mean (smoothed)', color='red')
    
    plt.xlabel("Total Timesteps")
    plt.ylabel("Rollout Mean Episode Reward")
    plt.title("Training Progress (Diagnostic Run)")
    plt.legend()
    plt.grid(True)
    
    output_path = r"d:\UAV Dynamic Routing\uav-dynamic-routing\diagnostic_learning_curve.png"
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    
    # Simple analysis logic for output text
    last_idx = len(ep_rew_means) - 1
    start_mean = ep_rew_means[0]
    peak_mean = max(ep_rew_means)
    end_mean = ep_rew_means[-1]
    
    print(f"\nStats: Start={start_mean}, Peak={peak_mean}, End={end_mean}")
else:
    print("No data extracted!")
