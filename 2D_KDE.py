import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import glob
import os
import re


def load_granulometry_matrix(m_file_path):
    """
    Parses a text-based .m file containing a printed Granulometry(:,:) matrix.
    """
    rows = []
    reading_matrix = False

    with open(m_file_path, "r") as f:
        for line in f:
            line = line.strip()

            # Detect start
            if "Granulometry" in line:
                reading_matrix = True
                continue

            if reading_matrix:
                # Skip header lines with indices
                if re.match(r"^\d+(\s+\d+)*$", line):
                    continue

                # Extract numbers (floats or ints)
                parts = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                if parts:
                    rows.append([float(x) for x in parts])

    if not rows:
        raise RuntimeError(f"No matrix data found in {m_file_path}")

    return np.array(rows)


def granulometry_to_samples(G):
    shape_dim, size_dim = G.shape

    shape_idx, size_idx = np.meshgrid(
        np.arange(shape_dim), np.arange(size_dim), indexing="ij"
    )

    shape_flat = shape_idx.ravel()
    size_flat = size_idx.ravel()
    weights_flat = G.ravel()

    mask = weights_flat > 0

    coords = np.vstack([size_flat[mask], shape_flat[mask]])
    weights = weights_flat[mask]

    return coords, weights


def compute_kde_2d(coords, weights):
    return st.gaussian_kde(coords, weights=weights)


def save_kde_2d(kde, G, out_path, title="KDE"):
    shape_dim, size_dim = G.shape

    X, Y = np.meshgrid(
        np.linspace(0, shape_dim - 1, 200),
        np.linspace(0, size_dim - 1, 200)
    )

    grid_coords = np.vstack([X.ravel(), Y.ravel()])
    Z = kde(grid_coords).reshape(X.shape)

    plt.figure(figsize=(7, 6))
    plt.imshow(
        Z,
        origin='lower',
        extent=[0, shape_dim - 1, 0, size_dim - 1],
        aspect='auto'
    )
    plt.colorbar(label="Density")
    plt.xlabel("Shape index")
    plt.ylabel("Size index")
    plt.title(title)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_histogram(G, out_path, title="Histogram"):
    """
    Plots the original binned granulometry histogram from the .m file.
    """
    plt.figure(figsize=(7, 6))
    plt.imshow(
        G,
        origin='lower',
        cmap='viridis',
        aspect='auto'
    )
    plt.colorbar(label="Count")
    plt.xlabel("Size index")
    plt.ylabel("Shape index")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def process_folder(folder_path, kde_out_dir, hist_out_dir):
    """
    Produces KDE and histogram images for each .m granulometry file.
    """
    m_files = sorted(glob.glob(os.path.join(folder_path, "*.m")))
    if not m_files:
        raise RuntimeError(f"No .m files found in {folder_path}")

    os.makedirs(kde_out_dir, exist_ok=True)
    os.makedirs(hist_out_dir, exist_ok=True)
    print(f"Found {len(m_files)} .m files. Saving KDEs to {kde_out_dir} and histograms to {hist_out_dir} ...")

    for f in m_files:
        print(f"Processing {f} ...")

        try:
            G = load_granulometry_matrix(f)
        except Exception as e:
            print(f"Skipping {f}: {e}")
            continue

        coords, weights = granulometry_to_samples(G)
        kde = compute_kde_2d(coords, weights)

        base_name = os.path.splitext(os.path.basename(f))[0]

        kde_out_path = os.path.join(kde_out_dir, f"{base_name}_kde.png")
        save_kde_2d(kde, G, kde_out_path, title=f"{base_name} KDE")

        hist_out_path = os.path.join(hist_out_dir, f"{base_name}_hist.png")
        save_histogram(G, hist_out_path, title=f"{base_name} Histogram")


if __name__ == "__main__":
    ROOT_DIR = "xmaxtree/output/Spunta/HealthyLeaf_Rchannel"
    KDE_OUT_DIR = "kde_outputs/HealthyLeaf_Rchannel"
    HIST_OUT_DIR = "hist_outputs/HealthyLeaf_Rchannel"

    process_folder(ROOT_DIR, KDE_OUT_DIR, HIST_OUT_DIR)
