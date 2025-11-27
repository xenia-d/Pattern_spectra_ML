import os
import glob
import numpy as np
from itertools import combinations
from lvq.IAALVQ import IAALVQ
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import torch
import random
import argparse


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
    print(f"Confusion matrix saved to: {os.path.join(out_dir, f'{name}.png')}")

def load_granulometry_m_file(path):
    data = []
    with open(path, 'r') as f:
        lines = f.readlines()
    lines = lines[2:]  # skip headers
    for line in lines:
        line = line.strip()
        if not line:
            continue
        tokens = line.split()
        numbers = [float(x) for x in tokens[1:]]
        data.append(numbers)
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
        n_val = max(1, int(round(len(imgs) * validation_fraction)))  # ensure at least 1 sample
        imgs_val = imgs[I[:n_val]]
        labels_val = labels[I[:n_val]]
        imgs_train = imgs[I[n_val:]]
        labels_train = labels[I[n_val:]]
        yield imgs_train, labels_train, imgs_val, labels_val


def main():
    np.random.seed(12)
    torch.manual_seed(12)

    parser = argparse.ArgumentParser(description="Potato variant")
    parser.add_argument("--variant", type=str, default="Spunta",
                        help="Potato variant folder to use (e.g., Spunta, Mondial, Fontane, Rudolph)")
    args = parser.parse_args()



    ROOT_DIR = f"xmaxtree/output/{args.variant}"
    xval_count = 5
    xval_fraction = 0.2
    iterations = 3  
    label_map = {"Healthy": "h", "Unhealthy": "uh"}
    channel_pool = ["R", "G", "B", "H", "S", "V"]
    
    # Generate all single, pairwise, and 3-channel combinations
    all_combinations = []
    for r in range(1, 4):
        all_combinations.extend(combinations(channel_pool, r))

    results = []

    for combo in all_combinations:
        print(f"\n=== Running experiment for channels: {combo} ===")
        
        iter_f1_list = []
        iter_acc_list = []
        
        for it in range(iterations):
            print(f"\n--- Iteration {it+1}/{iterations} ---")
            all_features = []
            all_labels = []

            # Load features for Healthy and Unhealthy
            for label_name, prefix in label_map.items():
                # Load files per channel
                files_dict = {}
                for ch in channel_pool:
                    files_dict[ch] = sorted(glob.glob(os.path.join(ROOT_DIR, f"{label_name}Leaf_{ch}channel", f"{prefix}{ch}*.m")))
                n_samples = len(files_dict[combo[0]])
                feats = []
                for i in range(n_samples):
                    file_dict = {ch: files_dict[ch][i] for ch in combo}
                    feat = create_feature_vector(file_dict, combo)
                    if feat is not None:
                        feats.append(feat)
                feats = np.array(feats)
                labels = np.array([0 if label_name=="Healthy" else 1]*len(feats))
                all_features.append(feats)
                all_labels.append(labels)

            # Combine datasets
            if any(len(f)==0 for f in all_features):
                print(f"Skipping combination {combo}, no valid features for one of the classes.")
                continue

            X = np.vstack(all_features)
            y = np.concatenate(all_labels)

            splitter = random_splitter_gen(X, y, xval_fraction)
            f1_list, acc_list, conf_list = [], [], []

            for fold in range(xval_count):
                x_train, y_train, x_test, y_test = next(splitter)
                model = IAALVQ(max_iter=100, prototypes_per_class=2, omega_rank=400, seed=59,
                                regularization=1e-5, omega_locality='PW', filter_bank=None,
                                block_eye=False, norm=False, correct_imbalance=True)
                x_train = np.array(x_train)
                x_test = np.array(x_test)
                y_train = np.array(y_train, dtype=np.int64)
                y_test = np.array(y_test, dtype=np.int64)

                model.fit(x_train, y_train)
                y_pred_test = model.predict(x_test)

                f1_test = f1_score(y_test, y_pred_test, average='weighted')
                acc_test = accuracy_score(y_test, y_pred_test)
                conf = confusion_matrix(y_test, y_pred_test, normalize='true')

                f1_list.append(f1_test)
                acc_list.append(acc_test)
                conf_list.append(conf)

                print(f"Fold {fold+1} - F1: {f1_test:.4f}, Acc: {acc_test:.4f}")

            iter_f1_list.append(np.mean(f1_list))
            iter_acc_list.append(np.mean(acc_list))

        # Average across iterations
        avg_f1 = np.mean(iter_f1_list)
        avg_acc = np.mean(iter_acc_list)

        combo_name = "_".join(combo)
        results.append((combo_name, avg_f1, avg_acc))
        print(f"\nChannels {combo} - Avg F1 over {iterations} iterations: {avg_f1:.4f}, Avg Acc: {avg_acc:.4f}")

    # Rank results by test F1
    ranked_results = sorted(results, key=lambda x: x[1], reverse=True)
    print(f"\n===== RANKED CHANNEL COMBINATIONS (Best -> Worst) for variant {args.variant} =====")
    for rank, (combo_name, f1, acc) in enumerate(ranked_results, 1):
        print(f"{rank}. {combo_name}: F1 = {f1:.4f}, Accuracy = {acc:.4f}")


if __name__ == "__main__":
    main()
