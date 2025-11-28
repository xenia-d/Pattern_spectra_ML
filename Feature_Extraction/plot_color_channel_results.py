import os
import pickle
import matplotlib.pyplot as plt
import numpy as np


variants = ["Spunta", "Mondial"] 
pkl_dir = "saved_color_channel_combo_results"
output_dir = "Plots"
os.makedirs(output_dir, exist_ok=True)

for variant in variants:
    pkl_file = os.path.join(pkl_dir, f"{variant}_channel_results.pkl")
    
    if not os.path.exists(pkl_file):
        print(f"PKL file not found for variant {variant}: {pkl_file}")
        continue

    with open(pkl_file, "rb") as f:
        results = pickle.load(f)

    combo_names = [r[0] for r in results]
    avg_f1_scores = [r[1] for r in results]

    # Sort by F1 descending
    sorted_indices = np.argsort(avg_f1_scores)[::-1]
    combo_names_sorted = [combo_names[i] for i in sorted_indices]
    avg_f1_scores_sorted = [avg_f1_scores[i] for i in sorted_indices]

    plt.figure(figsize=(10, 8))
    y_pos = np.arange(len(combo_names_sorted))
    bars = plt.barh(y_pos, avg_f1_scores_sorted, color='teal')
    plt.yticks(y_pos, combo_names_sorted)
    plt.xlabel("Average F1 Score")
    plt.xlim(0, 1)
    plt.title(f"Average F1 Score per Channel Combination — {variant}")

    for bar, f1 in zip(bars, avg_f1_scores_sorted):
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, f"{f1:.3f}",
                 va='center', fontsize=8)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{variant}_avg_f1_barh.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Plot saved for {variant}: {plot_path}")
