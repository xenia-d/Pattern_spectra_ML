import os
import glob
import numpy as np
from itertools import combinations
from lvq.IAALVQ import IAALVQ
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
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

def create_feature_vector(file_dict, channels):
    arrays = []
    for ch in channels:
        arr = np.array(load_granulometry_m_file(file_dict[ch]))[:10, :10].flatten()
        if np.max(arr) == 0:
            return None
        arrays.append(arr)
    return np.concatenate(arrays, axis=0)

def random_splitter_gen(imgs, labels, validation_fraction):
    imgs = np.array(imgs)
    labels = np.array(labels)
    while True:
        I = np.random.permutation(len(imgs))
        n_val = max(1, int(round(len(imgs) * validation_fraction)))
        yield imgs[I[n_val:]], labels[I[n_val:]], imgs[I[:n_val]], labels[I[:n_val]]

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

def run_iteration(X_combo, y_combo, xval_count, xval_fraction, seed):
    np.random.seed(seed)
    splitter = random_splitter_gen(X_combo, y_combo, xval_fraction)
    f1_list, acc_list, conf_list = [], [], []
    for fold in range(xval_count):
        x_train, y_train, x_val, y_val = next(splitter)
        f1, acc, conf = evaluate_fold(x_train, y_train, x_val, y_val)
        f1_list.append(f1)
        acc_list.append(acc)
        conf_list.append(conf)
    return np.mean(f1_list), np.mean(acc_list), np.mean(conf_list, axis=0)


def main():
    np.random.seed(12)
    torch.manual_seed(12)

    parser = argparse.ArgumentParser(description="Potato variant")
    parser.add_argument("--variant", type=str, default="Spunta",
                        help="Potato variant folder (Spunta, Mondial, Fontane, Rudolph)")
    args = parser.parse_args()

    ROOT_DIR = f"xmaxtree/output/{args.variant}"
    xval_count = 5
    xval_fraction = 0.2
    iterations = 3
    label_map = {"Healthy": "h", "Unhealthy": "uh"}
    channel_pool = ["R", "G", "B", "H", "S", "V"]

    # Preload all features
    print("Preloading all features...")
    preloaded_features = {}
    for label_name, prefix in label_map.items():
        preloaded_features[label_name] = {}
        for ch in channel_pool:
            files = sorted(glob.glob(os.path.join(ROOT_DIR, f"{label_name}Leaf_{ch}channel", f"{prefix}{ch}*.m")))
            feats = []
            for f in files:
                arr = np.array(load_granulometry_m_file(f))[:10, :10].flatten()
                if np.max(arr) == 0:  # skip zero vectors
                    continue
                feats.append(arr + 1e-12)  # tiny epsilon to avoid exact zeros
            preloaded_features[label_name][ch] = np.array(feats)
    print("Feature preload complete.")

    # Split into held-out test set (80-20)
    for label_name in label_map:
        n_samples = preloaded_features[label_name][channel_pool[0]].shape[0]
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        n_test = max(1, int(round(n_samples * xval_fraction)))
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]
        for ch in channel_pool:
            feats = preloaded_features[label_name][ch]
            preloaded_features[label_name][ch+"_train"] = feats[train_idx]
            preloaded_features[label_name][ch+"_test"] = feats[test_idx]

    # Generate all channel combinations
    from itertools import combinations
    all_combinations = []
    for r in range(1,4):
        all_combinations.extend(combinations(channel_pool, r))

    results = []
    best_f1 = -1
    best_combo_name = None

    for combo in all_combinations:
        print(f"\n=== Running experiment for channels: {combo} ===")
        # Build train feature set
        X_combo = np.vstack([
            np.hstack([preloaded_features[label][ch+"_train"] for ch in combo])
            for label in label_map
        ])
        y_combo = np.concatenate([
            np.array([0 if l=="Healthy" else 1]*preloaded_features[l][combo[0]+"_train"].shape[0])
            for l in label_map
        ])

        # Run sequential iterations
        iter_f1_scores, iter_acc_scores, iter_conf_matrices = [], [], []
        for seed in range(iterations):
            np.random.seed(seed)
            splitter = random_splitter_gen(X_combo, y_combo, xval_fraction)
            f1_list, acc_list, conf_list = [], [], []
            for fold in range(xval_count):
                x_train, y_train, x_val, y_val = next(splitter)
                f1, acc, conf = evaluate_fold(x_train, y_train, x_val, y_val)
                f1_list.append(f1)
                acc_list.append(acc)
                conf_list.append(conf)
            iter_f1_scores.append(np.mean(f1_list))
            iter_acc_scores.append(np.mean(acc_list))
            iter_conf_matrices.append(np.mean(conf_list, axis=0))

        avg_f1 = np.mean(iter_f1_scores)
        avg_acc = np.mean(iter_acc_scores)
        avg_conf = np.mean(iter_conf_matrices, axis=0)

        results.append({
            "combo": "_".join(combo),
            "f1_scores": iter_f1_scores,
            "acc_scores": iter_acc_scores,
            "avg_f1": avg_f1,
            "avg_acc": avg_acc
        })

        print(f"Combo {combo}: Avg F1={avg_f1:.4f}, Avg Acc={avg_acc:.4f}")

        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_combo_name = combo

    print("\n===== BEST COMBINATION =====")
    print(f"{best_combo_name} — Avg F1={best_f1:.4f}")

    # Retrain best combo on full train set and evaluate on held-out test set
    X_best_train = np.vstack([
        np.hstack([preloaded_features[label][ch+"_train"] for ch in best_combo_name])
        for label in label_map
    ])
    y_best_train = np.concatenate([
        np.array([0 if l=="Healthy" else 1]*preloaded_features[l][best_combo_name[0]+"_train"].shape[0])
        for l in label_map
    ])
    X_best_test = np.vstack([
        np.hstack([preloaded_features[label][ch+"_test"] for ch in best_combo_name])
        for label in label_map
    ])
    y_best_test = np.concatenate([
        np.array([0 if l=="Healthy" else 1]*preloaded_features[l][best_combo_name[0]+"_test"].shape[0])
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
    final_conf = confusion_matrix(y_best_test, y_pred_test, normalize='true')

    print("\n===== FINAL TEST SET RESULTS =====")
    print(f"Test Weighted F1: {final_f1:.4f}, Accuracy: {final_acc:.4f}")
    print("Test Confusion Matrix (normalized):")
    print(final_conf)

    os.makedirs("Saved_Results", exist_ok=True)
    with open(os.path.join("Saved_Results", f"{args.variant}_best_model.pkl"), "wb") as f:
        pickle.dump(final_model, f)
    with open(os.path.join("Saved_Results", f"{args.variant}_channel_results.pkl"), "wb") as f:
        pickle.dump(results, f)
    save_confusion_matrix(final_conf, out_dir="Saved_Results", name=f"{args.variant}_best_conf_matrix")

if __name__ == "__main__":
    main()
