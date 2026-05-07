import numpy as np
import matplotlib.pyplot as plt
import os

# --- Configuration ---
# Update this list with the dataset sizes you actually trained
n_sizes = [256, 512, 1024, 2048] 

img_size = 32
n_base = 32
optim = 'Adam'
lr = 0.0001
index = 0
base_save_path = '../../Saves/final_models'  # Update this path if your saves are located elsewhere

# Colors for different n sizes
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

fig, ax1 = plt.subplots(figsize=(10, 6))
# Create a secondary y-axis for f_mem
ax2 = ax1.twinx()

for idx, n in enumerate(n_sizes):
    # Calculate the batch size used for this run
    batch_size = min(512, n)
    
    # Construct the exact folder name
    model_name = f"ISIC{img_size}_{n}_{n_base}_{optim}_{batch_size}_{lr:.4f}_index{index}"
    
    # File paths
    fid_file = os.path.join(base_save_path, model_name, 'FID', 'FID_1.txt')
    fmem_file = os.path.join(base_save_path, model_name, 'Memorization', 'fraction_memorized.txt')
    
    color = colors[idx % len(colors)]
    
    # --- 1. Plot FID (Generalization) ---
    if os.path.exists(fid_file):
        # Load tau and FID score
        data_fid = np.loadtxt(fid_file)
        if data_fid.ndim > 1 and len(data_fid) > 0:
            tau_fid = data_fid[:, 0]
            fid_scores = data_fid[:, 1]
            
            # Filter out failed FID computations (-1)
            valid = fid_scores >= 0
            
            ax1.plot(tau_fid[valid], fid_scores[valid], linestyle='--', marker='o', 
                     color=color, label=f'FID (n={n})', markersize=4)
    else:
        print(f"Warning: Could not find FID file for n={n}")

    # --- 2. Plot f_mem (Memorization) ---
    if os.path.exists(fmem_file):
        # Load tau, f_mem, std, lower, upper
        data_fmem = np.loadtxt(fmem_file)
        if data_fmem.ndim > 1 and len(data_fmem) > 0:
            tau_fmem = data_fmem[:, 0]
            fmem_scores = data_fmem[:, 1]
            
            ax2.plot(tau_fmem, fmem_scores, linestyle='-', marker='s', 
                     color=color, label=f'f_mem (n={n})', markersize=5, linewidth=2)
    else:
        print(f"Warning: Could not find f_mem file for n={n}")

# --- Formatting the Plot ---
# Make the x-axis logarithmic because training steps span orders of magnitude
ax1.set_xscale('log')

ax1.set_xlabel(r'Training Steps ($\tau$)', fontsize=14)
ax1.set_ylabel('FID Score (Lower is better)', fontsize=14)
ax2.set_ylabel(r'Memorization Fraction $f_{mem}$ (%)', fontsize=14)

ax1.grid(True, which="both", ls="-", alpha=0.2)

# Combine legends from both axes
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', 
           bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=12)

plt.title('Generalization vs Memorization Timescales on ISIC', fontsize=16)
plt.tight_layout()

# Save the plot
output_dir = '../../Saves/final_models/Plots'
os.makedirs(output_dir, exist_ok=True)
output_filename = 'Phase_Diagram_Unconditioned.png'
plt.savefig(os.path.join(output_dir, output_filename), dpi=300, bbox_inches='tight')
print(f"\nPlot successfully saved as: {os.path.join(output_dir, output_filename)}")