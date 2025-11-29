import os
import pickle
import matplotlib.pyplot as plt
import numpy as np


def plot_ranked_channel_combos(variants, pkl_dir="Channel_Combo_Results", output_dir="Plots"):
    os.makedirs(output_dir, exist_ok=True)

    for variant in variants:
        pkl_file = os.path.join(pkl_dir, f"{variant}_channel_results.pkl")


        print(f"Loading {pkl_file} ...")
        with open(pkl_file, "rb") as f:
            results = pickle.load(f)

        combo_names = [r["combo"] for r in results]
        avg_f1_scores = [r["avg_f1"] for r in results]
        std_f1_scores = [np.std(r["f1_scores"]) for r in results]

        sorted_idx = np.argsort(avg_f1_scores)[::-1]
        combo_sorted = [combo_names[i] for i in sorted_idx]
        avg_f1_sorted = [avg_f1_scores[i] for i in sorted_idx]
        std_f1_sorted = [std_f1_scores[i] for i in sorted_idx]


        plt.figure(figsize=(10, 8))
        y_pos = np.arange(len(combo_sorted))

        plt.barh(
            y_pos,
            avg_f1_sorted,
            xerr=std_f1_sorted,
            capsize=4,
            color='teal',
            alpha=0.8
        )


        plt.gca().invert_yaxis()

        plt.yticks(y_pos, combo_sorted)
        plt.xlabel("Average F1 Score")
        plt.xlim(0, 1)
        plt.title(f"{variant} — Ranked Channel Combinations (F1 ± STD)")


        for y, f1 in zip(y_pos, avg_f1_sorted):
            plt.text(f1 + 0.01, y, f"{f1:.3f}", va='center', fontsize=8)

        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"{variant}_avgF1_std_ranked.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()


def plot_test_f1_scores(test_f1_dict, output_dir="Plots"):
    os.makedirs(output_dir, exist_ok=True)

    variants = list(test_f1_dict.keys())
    scores = list(test_f1_dict.values())

    plt.figure(figsize=(6, 5))
    x_pos = np.arange(len(variants))

    plt.bar(x_pos, scores, alpha=0.85)

    plt.xticks(x_pos, variants)
    plt.ylim(0, 1)
    plt.ylabel("Test Weighted F1")
    plt.title("Test F1 Scores by Variant")

    for x, f1 in zip(x_pos, scores):
        plt.text(x, f1 + 0.01, f"{f1:.4f}", ha='center')

    out_path = os.path.join(output_dir, "Test_F1_Scores.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()



if __name__ == "__main__":

    variants = ["Rudolph", "Mondial"]
    plot_ranked_channel_combos(variants)

    test_f1_scores = {
        "Mondial": 0.9071,
        "Rudolph": 0.9406,
        # "Spunta": 0.xxx,
        # "Fontane": 0.xxx
    }

    # Run test F1 score plotting
    plot_test_f1_scores(test_f1_scores)
