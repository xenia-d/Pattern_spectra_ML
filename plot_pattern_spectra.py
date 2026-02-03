import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm


def load_granulometry_m_file(path):
    data = []
    with open(path, 'r') as f:
        lines = f.readlines()[2:]  # skip the first two header lines
    for line in lines:
        tokens = line.strip().split()
        if tokens:
            data.append([float(x) for x in tokens[1:]])  # skip column 0
    return np.array(data)


def plot_pattern_spectra(data, title="Pattern Spectra", threshold_visible=2000):
    s_bins, r_bins = data.shape

    fig, ax = plt.subplots(figsize=(8, 6))

    # PowerNorm to emphasize lower values
    norm = PowerNorm(gamma=0.5, vmin=threshold_visible, vmax=data.max())

    im = ax.imshow(data, cmap="hot", norm=norm)

    ax.set_xlabel("r (Size)")
    ax.set_ylabel("s (Shape)")
    ax.set_title(title)

    ax.set_xticks(np.arange(r_bins))
    ax.set_yticks(np.arange(s_bins))

    ax.invert_yaxis()

    cbar = plt.colorbar(im)
    cbar.set_label("Value ")

    max_val = data.max()

    for i in range(s_bins):
        for j in range(r_bins):
            val = data[i, j]
            if val > 0:  # show only non-zero cells
                ax.text(
                    j, i, f"{int(val)}",
                    ha="center", va="center",
                    fontsize=7,
                    color="black" if val < max_val * 0.6 else "white"
                )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    path = r"xmaxtree/output/Spunta/HealthyLeaf_Gchannel/hG_3081_55_healthybox_Gpatch_17.m"
    data = load_granulometry_m_file(path)
    plot_pattern_spectra(data, title="Granulometry Pattern Spectra", threshold_visible=10)
