import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

EVAL_MODE = "deletion"  

VARIANTS = ["Rudolph", "Mondial", "Spunta", "Fontane"] 
RESULTS_DIR = "Bin_Analysis_Results"     # folder with PKLs
PLOT_DIR = "Plots/bin_analysis/Deletion"   # folder where figures will be saved

COLOR_MAP = {
    "R": "#E96060",     
    "B": "#4EA7FF",      
    "G": "#98D798",      
    "H": "#EC7FCA",      
    "V": "#F3BA6B",      
    "S": "#792468",      
}
def find_channels_for_variant(variant):
    channels = []
    for fname in os.listdir(RESULTS_DIR):
        if fname.startswith(f"{variant}_bin_analysis_") and fname.endswith(".pkl"):
            ch = fname.replace(f"{variant}_bin_analysis_", "").replace(".pkl", "")
            channels.append(ch)

    preferred_order = ["R", "G", "B", "H", "S", "V"]
    channels_sorted = sorted(
        channels,
        key=lambda c: preferred_order.index(c) if c in preferred_order else len(preferred_order)
    )
    return channels_sorted


def load_channel_results(variant, channel):
    path = os.path.join(RESULTS_DIR, f"{variant}_bin_analysis_{channel}.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def collect_variant_data(variant, channels):
    data = {k:{} for k in BIN_GROUPS}

    for ch in channels:
        results = load_channel_results(variant, ch)
        if results is None:
            continue

        for entry in results:
            if "f1_mean_across_iters" not in entry:
                continue

            kind = entry["kind"]
            if kind not in data:
                continue

            data[kind][ch] = (
                entry["f1_mean_across_iters"],
                entry["f1_std_across_iters"],
            )

    return data

def plot_variant(variant):
    os.makedirs(PLOT_DIR, exist_ok=True)

    channels = find_channels_for_variant(variant)
    data = collect_variant_data(variant, channels)

    x = np.arange(len(BIN_GROUPS))
    num_channels = len(channels)
    width = 0.8 / num_channels

    plt.figure(figsize=(14,7))
    for i, ch in enumerate(channels):
        means = []
        for kind in BIN_GROUPS:
            if ch in data[kind]:
                m, s = data[kind][ch]
                means.append(m)
            else:
                means.append(np.nan)

        # offset each channel's bar horizontally
        offset = (i - num_channels/2) * width + width/2

        # plot bar with the color from COLOR_MAP 
        plt.bar(
            x + offset,
            means,
            width=width,
            capsize=4,
            label=f"{ch} channel",
            alpha=0.8,
            color=COLOR_MAP.get(ch, 'black')
        )

    # annotate bars with mean f1 only for the highest mean F1 per channel
    for i, ch in enumerate(channels):
        means = []
        for kind in BIN_GROUPS:
            if ch in data[kind]:
                m, s = data[kind][ch]
                means.append(m)
            else:
                means.append(np.nan)

        max_mean_idx = np.nanargmax(means)
        max_mean = means[max_mean_idx]

        # make annotation appear above the bar, with a small margin above
        offset = (i - num_channels/2) * width + width/2
        plt.text(
            x[max_mean_idx] + offset,
            max_mean + 0.03,
            f"{max_mean:.3f}",
            ha="center",
            fontsize=8
        )

    print("mean F1 and STD per best bin group per channel:", {ch: (data[BIN_GROUPS[np.nanargmax([data[kind][ch][0] if ch in data[kind] else np.nan for kind in BIN_GROUPS])]][ch]) for ch in channels})

    bin_labels = [label.replace("_", "\n") for label in BIN_GROUPS]
    
    plt.xticks(x, bin_labels, fontsize=14)
    plt.ylabel("F1 Score (mean ± std)")
    if EVAL_MODE == "deletion":
        plt.title(f"Pattern Spectra Bin Group Analysis – {variant} - Deletion")
    else:
        plt.title(f"Pattern Spectra Bin Group Analysis – {variant} - Insertion")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(PLOT_DIR, f"{variant}_bin_group_barplot.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Bar plot for {variant} ---> {save_path}")


if __name__ == "__main__":

    if EVAL_MODE == "deletion":
        PLOT_DIR = "Plots/bin_analysis/Deletion"
        RESULTS_DIR = "Bin_Analysis_Results/Deletion"
    else:
        PLOT_DIR = "Plots/bin_analysis/Insertion"
        RESULTS_DIR = "Bin_Analysis_Results/Insertion"


    # Order of subsets on x-axis
    BIN_GROUPS = [
        "small_sizes",
        "large_sizes",
        "elongated_shapes",
        "compact_shapes",
        "elongated_small",
        "elongated_large",
        "compact_small",
        "compact_large",
    ]


    os.makedirs(PLOT_DIR, exist_ok=True)

    for variant in VARIANTS:
        plot_variant(variant)

