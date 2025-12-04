import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


VARIANTS = ["Rudolph", "Mondial"] 
RESULTS_DIR = "Bin_Analysis_Results"     # folder with PKLs
PLOT_DIR = "Plots"   # folder where figures will be saved

# Order of subsets on x-axis
EXPECTED_KINDS = [
    "small_sizes",
    "large_sizes",
    "elongated_shapes",
    "compact_shapes",
    "elongated_small",
    "elongated_large",
    "compact_small",
    "compact_large",
]


def find_channels_for_variant(variant):
    channels = []
    for fname in os.listdir(RESULTS_DIR):
        if fname.startswith(f"{variant}_bin_analysis_") and fname.endswith(".pkl"):
            ch = fname.replace(f"{variant}_bin_analysis_", "").replace(".pkl", "")
            channels.append(ch)
    return sorted(channels)


def load_channel_results(variant, channel):
    path = os.path.join(RESULTS_DIR, f"{variant}_bin_analysis_{channel}.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def collect_variant_data(variant, channels):
    data = {k:{} for k in EXPECTED_KINDS}

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
    if not channels:
        print(f"[WARN] No channel PKLs found for {variant}. Skipping.")
        return

    print(f"[INFO] Variant {variant} → channels found: {channels}")

    data = collect_variant_data(variant, channels)

    x = np.arange(len(EXPECTED_KINDS))
    num_channels = len(channels)
    width = 0.8 / num_channels

    plt.figure(figsize=(14,7))
    for i, ch in enumerate(channels):
        means = []
        stds = []
        for kind in EXPECTED_KINDS:
            if ch in data[kind]:
                m, s = data[kind][ch]
                means.append(m)
                stds.append(s)
            else:
                means.append(np.nan)
                stds.append(0)

        # offset each channel's bar horizontally
        offset = (i - num_channels/2) * width + width/2

        plt.bar(
            x + offset,
            means,
            width=width,
            yerr=stds,
            capsize=4,
            label=f"{ch} channel",
            alpha=0.8
        )

    plt.xticks(x, EXPECTED_KINDS, rotation=45, ha="right")
    plt.ylabel("F1 Score (mean ± std)")
    plt.title(f"Pattern Spectra Bin Group Analysis – {variant}")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(PLOT_DIR, f"{variant}_bin_group_barplot.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[SAVED] Bar plot for {variant} → {save_path}")


if __name__ == "__main__":
    os.makedirs(PLOT_DIR, exist_ok=True)

    for variant in VARIANTS:
        plot_variant(variant)

    print("\nDone. All bar plots saved to /Plots/")
