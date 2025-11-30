import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

import glob
import numpy as np
from itertools import combinations
from lvq.IAALVQ import IAALVQ
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import torch
import pickle
import argparse
from joblib import Parallel, delayed

def save_confusion_matrix(conf_matrix, out_dir="Saved_Results", name="confusion_matrix"):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(6,5))
    plt.imshow(conf_matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (Normalized)")
    plt.colorbar()
    classes = ["Healthy", "Unhealthy"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, f"{conf_matrix[i,j]:.2f}",
                     horizontalalignment="center",
                     color="white" if conf_matrix[i,j] > thresh else "black")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}.png"), dpi=300)
    plt.close()


def load_granulometry_m_file(path):
    data = []
    with open(path, 'r') as f:
        lines = f.readlines()[2:]  # skip headers
    for line in lines:
        tokens = line.strip().split()
        if tokens:
            data.append([float(x) for x in tokens[1:]])
    return np.array(data)


def evaluate_fold(x_train, y_train, x_test, y_test):
    model = IAALVQ(
        max_iter=100,
        prototypes_per_class=2,
        omega_rank=400,
        seed=59,
        regularization=1e-5,
        omega_locality='PW',
        filter_bank=None,
        block_eye=False,
        norm=False,
        correct_imbalance=True
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    acc = accuracy_score(y_test, y_pred)
    conf = confusion_matrix(y_test, y_pred, normalize='true')
    return f1, acc, conf


def evaluate_combo(combo, preloaded_features, label_map, xval_count, iterations):

    # Build training dataset for this combo
    X_combo = np.vstack([
        np.hstack([preloaded_features[label][ch + "_train"] for ch in combo])
        for label in label_map
    ])

    y_combo = np.concatenate([
        np.array([0 if l == "Healthy" else 1] *
                 preloaded_features[l][combo[0] + "_train"].shape[0])
        for l in label_map
    ])

    iter_f1_scores = []
    iter_acc_scores = []
    iter_conf_mats = []

    for seed in range(iterations):

        kf = KFold(n_splits=xval_count, shuffle=True, random_state=seed)

        fold_f1 = []
        fold_acc = []
        fold_conf = []

        for train_idx, val_idx in kf.split(X_combo):
            x_train, x_val = X_combo[train_idx], X_combo[val_idx]
            y_train, y_val = y_combo[train_idx], y_combo[val_idx]

            f1, acc, conf = evaluate_fold(x_train, y_train, x_val, y_val)

            fold_f1.append(f1)
            fold_acc.append(acc)
            fold_conf.append(conf)

        iter_f1_scores.append(np.mean(fold_f1))
        iter_acc_scores.append(np.mean(fold_acc))
        iter_conf_mats.append(np.mean(fold_conf, axis=0))

    avg_f1 = np.mean(iter_f1_scores)
    avg_acc = np.mean(iter_acc_scores)
    avg_conf = np.mean(iter_conf_mats, axis=0)

    return {
        "combo": "_".join(combo),
        "avg_f1": avg_f1,
        "avg_acc": avg_acc,
        "f1_scores": iter_f1_scores,
        "acc_scores": iter_acc_scores,
        "conf_mat": avg_conf
    }

def main():
    np.random.seed(12)
    torch.manual_seed(12)

    parser = argparse.ArgumentParser(description="Potato variant")
    parser.add_argument("--variant", type=str, default="Spunta",
                        help="Potato variant folder (Spunta, Mondial, Fontane, Rudolph)")
    args = parser.parse_args()

    ROOT_DIR = os.path.join(PROJECT_ROOT, "xmaxtree", "output", args.variant)

    # Fontane is large → fewer CV folds
    xval_count = 3 if args.variant.lower() == "fontane" else 5
    xval_fraction = 0.2
    iterations = 3

    label_map = {"Healthy": "h", "Unhealthy": "uh"}
    channel_pool = ["R", "G", "B", "H", "S", "V"]

    print("Preloading all features...")

    preloaded_features = {}
    for label_name, prefix in label_map.items():
        preloaded_features[label_name] = {}
        for ch in channel_pool:
            files = sorted(glob.glob(os.path.join(
                ROOT_DIR, f"{label_name}Leaf_{ch}channel", f"{prefix}{ch}*.m")))
            feats = []
            for f in files:
                arr = np.array(load_granulometry_m_file(f))[:10, :10].flatten()
                if np.max(arr) == 0:
                    continue
                feats.append(arr)
            preloaded_features[label_name][ch] = np.array(feats)

    print("Feature preload complete.")

    # Split train/test once
    for label_name in label_map:
        n_samples = preloaded_features[label_name][channel_pool[0]].shape[0]
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        n_test = max(1, int(round(n_samples * xval_fraction)))
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

        for ch in channel_pool:
            feats = preloaded_features[label_name][ch]
            preloaded_features[label_name][ch + "_train"] = feats[train_idx]
            preloaded_features[label_name][ch + "_test"] = feats[test_idx]

    # Channel combinations
    all_combinations = []
    for r in range(1, 4):
        all_combinations.extend(combinations(channel_pool, r))

    print("\n=== Running all combinations in parallel ===")

    # PARALLEL EXECUTION
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(evaluate_combo)(
            combo, preloaded_features, label_map, xval_count, iterations
        )
        for combo in all_combinations
    )

    # Get best combo
    best_result = max(results, key=lambda r: r["avg_f1"])
    best_combo_name = best_result["combo"]
    best_f1 = best_result["avg_f1"]

    print("\n===== BEST COMBINATION =====")
    print(f"{best_combo_name} — Avg F1={best_f1:.4f}")

    # Evaluate best combo on test set
    best_combo_list = best_combo_name.split("_")

    X_best_train = np.vstack([
        np.hstack([preloaded_features[label][ch + "_train"] for ch in best_combo_list])
        for label in label_map
    ])
    y_best_train = np.concatenate([
        np.array([0 if l == "Healthy" else 1] *
                 preloaded_features[l][best_combo_list[0] + "_train"].shape[0])
        for l in label_map
    ])

    X_best_test = np.vstack([
        np.hstack([preloaded_features[label][ch + "_test"] for ch in best_combo_list])
        for label in label_map
    ])
    y_best_test = np.concatenate([
        np.array([0 if l == "Healthy" else 1] *
                 preloaded_features[l][best_combo_list[0] + "_test"].shape[0])
        for l in label_map
    ])

    final_model = IAALVQ(
        max_iter=100,
        prototypes_per_class=2,
        omega_rank=400,
        seed=59,
        regularization=1e-5,
        omega_locality='PW',
        filter_bank=None,
        block_eye=False,
        norm=False,
        correct_imbalance=True
    )
    final_model.fit(X_best_train, y_best_train)
    y_pred_test = final_model.predict(X_best_test)

    final_f1 = f1_score(y_best_test, y_pred_test, average='weighted')
    final_acc = accuracy_score(y_best_test, y_pred_test)
    final_precision = precision_score(y_best_test, y_pred_test, average='weighted')
    final_recall = recall_score(y_best_test, y_pred_test, average='weighted')
    final_conf = confusion_matrix(y_best_test, y_pred_test, normalize='true')

    print("\n===== FINAL TEST SET RESULTS =====")
    print(f"Test Weighted F1: {final_f1:.4f}")
    print(f"Test Accuracy: {final_acc:.4f}")
    print(f"Test Weighted Precision: {final_precision:.4f}")
    print(f"Test Weighted Recall: {final_recall:.4f}")
    print(final_conf)

    os.makedirs("Saved_Results", exist_ok=True)

    with open(os.path.join("Saved_Results", f"{args.variant}_best_model.pkl"), "wb") as f:
        pickle.dump(final_model, f)

    with open(os.path.join("Saved_Results", f"{args.variant}_channel_results.pkl"), "wb") as f:
        pickle.dump(results, f)

    save_confusion_matrix(final_conf, out_dir="Saved_Results",
                          name=f"{args.variant}_best_conf_matrix")


if __name__ == "__main__":
    main()
