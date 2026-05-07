import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Plot FID and f_mem for different skin types.")
    parser.add_argument("-n", "--num", type=int, required=True, help="Dataset size n (e.g. 1024)")
    parser.add_argument("-s", "--img_size", type=int, default=32, help="Image size")
    parser.add_argument("-W", "--nbase", type=int, default=32, help="Base filters")
    parser.add_argument("-O", "--optim", type=str, default="Adam", help="Optimizer")
    parser.add_argument("-LR", "--learning_rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("-B", "--batch_size", type=int, default=512, help="Batch size (max 512)")
    parser.add_argument("-i", "--index", type=int, default=0, help="Dataset split index")
    parser.add_argument("-D", "--dataset", type=str, default="ISIC_Conditioned", help="Dataset tag")
    parser.add_argument("-classes", nargs="+", type=int, required=True, help="List of skin type classes to plot (e.g. 2 5)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Reconstruct the exact folder name matching your training and evaluation scripts
    batch_size = min(args.batch_size, args.num)
    model_name = f"{args.dataset}{args.img_size}_{args.num}_{args.nbase}_{args.optim}_{batch_size}_{args.learning_rate:.4f}_index{args.index}"
    base_save_path = '../../Saves/final_models'
    
    # Distinct colors for clarity (Blue, Red, Green, Orange, Purple, Yellow)
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b']
    
    # Setup Figure and Axes
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    
    # Logarithmic X-axis for training steps
    ax1.set_xscale('log')
    ax1.set_xlabel(r'Training Steps ($\tau$)', fontsize=14)
    
    # Label axes
    ax1.set_ylabel('FID Score (Lower is better)', fontsize=14)
    ax2.set_ylabel(r'Memorization Fraction $f_{mem}$ (%)', fontsize=14)
    
    # FIX: Anchor both y-axes to exactly 0 at the bottom!
    ax2.set_ylim(0, 100)
    #ax1.set_ylim(bottom=0)
    
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    
    # Iterate over the requested skin types
    for idx, c in enumerate(args.classes):
        color = colors[idx % len(colors)]
        
        # File paths using the updated folder names
        fid_file = os.path.join(base_save_path, model_name, f'FID', f'FID_1_Class_{c}.txt')
        fmem_file = os.path.join(base_save_path, model_name, f'Memorization', f'fraction_memorized_Class_{c}.txt')
        
        # --- 1. Plot FID ---
        if os.path.exists(fid_file):
            data_fid = np.loadtxt(fid_file)
            if data_fid.ndim > 1 and len(data_fid) > 0:
                tau_fid = data_fid[:, 0]
                fid_scores = data_fid[:, 1]
                valid = fid_scores >= 0  # Filter out failed runs
                ax1.plot(tau_fid[valid], fid_scores[valid], linestyle='--', marker='o', 
                         color=color, label=f'FID (Type {c})', markersize=4)
        else:
            print(f"Warning: Could not find FID file for Type {c}: {fid_file}")

        # --- 2. Plot Fmem ---
        if os.path.exists(fmem_file):
            data_fmem = np.loadtxt(fmem_file)
            if data_fmem.ndim > 1 and len(data_fmem) > 0:
                tau_fmem = data_fmem[:, 0]
                fmem_scores = data_fmem[:, 1]
                ax2.plot(tau_fmem, fmem_scores, linestyle='-', marker='s', 
                         color=color, label=f'$f_{{mem}}$ (Type {c})', markersize=5, linewidth=2)
        else:
            print(f"Warning: Could not find f_mem file for Type {c}: {fmem_file}")

    # Combine legends into a single box at the bottom
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', 
               bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=12)

    plt.title(f'Generalization vs Memorization per Skin Type (n={args.num})', fontsize=16)
    plt.tight_layout()

    # Save to disk
    output_dir = '../../Saves/final_models/Plots'
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f'Phase_Diagram_Conditioned_n{args.num}.png'
    plt.savefig(os.path.join(output_dir, output_filename), dpi=300, bbox_inches='tight')
    print(f"\nPlot successfully saved as: {os.path.join(output_dir, output_filename)}")

if __name__ == '__main__':
    main()