import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

def visualize_lvq_channel_relevance(model, channel_names=["R","G","B","H","S","V"], save_path=None):
    """
    Visualizes only:
      1. Channel-level diagonal relevance
      2. Channel-level total relevance (sum of column norms)
    """

    omegas = model.omegas_
    nb_features = model.nb_features
    n_channels = len(channel_names)
    features_per_channel = nb_features // n_channels

    # ------------------------------
    # Diagonal relevance
    # ------------------------------
    diagonals = np.array([np.diag(omega) for omega in omegas])  # shape: (n_omegas, nb_features)
    mean_diag = diagonals.mean(axis=0)

    diag_channel_relevance = {}
    for i, ch in enumerate(channel_names):
        start = i * features_per_channel
        end = (i + 1) * features_per_channel
        diag_channel_relevance[ch] = np.sum(mean_diag[start:end])

    plt.figure(figsize=(8,5))
    plt.bar(diag_channel_relevance.keys(), diag_channel_relevance.values(), color='skyblue')
    plt.title("Channel-Level Diagonal Relevance")
    plt.ylabel("Sum of diag(Ω)")
    plt.grid(True, axis="y", alpha=0.3)
    if save_path:
        plt.savefig(save_path + "_diag_channel_relevance.png", dpi=300)
    plt.show()

    print("\n===== Diagonal Channel Relevance Summary =====")
    for ch, val in diag_channel_relevance.items():
        print(f"{ch}: {val:.4f}")

    # ------------------------------
    # Column-norm relevance
    # ------------------------------
    all_omegas = np.vstack(omegas)  # shape: (#omegas * omega_rank) x nb_features
    feature_relevance = np.linalg.norm(all_omegas, axis=0)

    channel_relevance = {}
    for i, ch in enumerate(channel_names):
        start = i * features_per_channel
        end = (i + 1) * features_per_channel
        channel_relevance[ch] = np.sum(feature_relevance[start:end])

    plt.figure(figsize=(8,5))
    plt.bar(channel_relevance.keys(), channel_relevance.values(), color='salmon')
    plt.title("Channel-Level Total Relevance (column norms)")
    plt.ylabel("Sum of feature norms")
    plt.grid(True, axis="y", alpha=0.3)
    if save_path:
        plt.savefig(save_path + "_channel_relevance.png", dpi=300)
    plt.show()

    print("\n===== Channel Relevance Summary =====")
    for ch, val in channel_relevance.items():
        print(f"{ch}: {val:.4f}")

    return diag_channel_relevance, channel_relevance


def main():
    MODEL_PATH = "Feature_Extraction/Saved_Models/Spunta_best_model-RGH.pkl" 
    SAVE_DIR = "Feature_Extraction/Plots/lvq_relevance"
    os.makedirs(SAVE_DIR, exist_ok=True)

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    print(f"Loaded model from: {MODEL_PATH}")
    print(f" - nb_features   = {model.nb_features}")
    print(f" - omega_rank    = {model.omega_rank}")
    print(f" - omegas count  = {len(model.omegas_)}")

    channel_names = ["R", "G", "B", "H", "S", "V"]
    visualize_lvq_channel_relevance(
        model,
        channel_names=channel_names,
        save_path=os.path.join(SAVE_DIR, "lvq")
    )

    print("\nSaved plots to:", SAVE_DIR)


if __name__ == "__main__":
    main()
