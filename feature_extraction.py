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
        n_val = max(1, int(round(len(imgs) * validation_fraction)))
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
                        help="Potato variant folder (Spunta, Mondial, Fontane, Rudolph)")
    args = parser.parse_args()

    ROOT_DIR = f"xmaxtree/output/{args.variant}"
    xval_count = 5
    xval_fraction = 0.2
    iterations = 3
    label_map = {"Healthy": "h", "Unhealthy": "uh"}
    channel_pool = ["R", "G", "B", "H", "S", "V"]

    # All 1, 2, 3-channel combos
    all_combinations = []
    for r in range(1, 4):
        all_combinations.extend(combinations(channel_pool, r))

    results = []

    best_f1 = -1
    best_model = None
    best_conf_matrix = None
    best_combo_name = None

    for combo in all_combinations:
        print(f"\n=== Running experiment for channels: {combo} ===")

        iter_f1_scores = []
        iter_acc_scores = []
        iter_conf_matrices = []  # averaged over folds

        for it in range(iterations):
            print(f"\n--- Iteration {it+1}/{iterations} ---")
            all_features = []
            all_labels = []

            # Load features for Healthy & Unhealthy
            for label_name, prefix in label_map.items():
                files_dict = {}
                for ch in combo:
                    files_dict[ch] = sorted(
                        glob.glob(os.path.join(ROOT_DIR,
                                               f"{label_name}Leaf_{ch}channel",
                                               f"{prefix}{ch}*.m"))
                    )
                n_samples = len(files_dict[combo[0]])
                feats = []
                for i in range(n_samples):
                    file_dict = {ch: files_dict[ch][i] for ch in combo}
                    feat = create_feature_vector(file_dict, combo)
                    if feat is not None:
                        feats.append(feat)

                feats = np.array(feats)
                labels = np.array([0 if label_name == "Healthy" else 1] * len(feats))

                all_features.append(feats)
                all_labels.append(labels)

            # Skip invalid combinations
            if any(len(f) == 0 for f in all_features):
                print(f"Skipping {combo}, missing samples.")
                continue

            X = np.vstack(all_features)
            y = np.concatenate(all_labels)

            splitter = random_splitter_gen(X, y, xval_fraction)
            f1_list, acc_list, conf_list = [], [], []

            for fold in range(xval_count):
                x_train, y_train, x_test, y_test = next(splitter)

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

                f1_list.append(f1)
                acc_list.append(acc)
                conf_list.append(conf)

                print(f"Fold {fold+1}: F1={f1:.4f}, Acc={acc:.4f}")

            iter_f1_scores.append(np.mean(f1_list))
            iter_acc_scores.append(np.mean(acc_list))
            iter_conf_matrices.append(np.mean(conf_list, axis=0))

        # Averages across iterations
        avg_f1 = np.mean(iter_f1_scores)
        avg_acc = np.mean(iter_acc_scores)
        avg_conf = np.mean(iter_conf_matrices, axis=0)

        combo_name = "_".join(combo)

        results.append({
            "combo": combo_name,
            "f1_scores": iter_f1_scores,
            "acc_scores": iter_acc_scores,
            "avg_f1": avg_f1,
            "avg_acc": avg_acc,
            "conf_matrices": iter_conf_matrices
        })

        # Track best combination
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_model = model
            best_conf_matrix = avg_conf
            best_combo_name = combo_name

        # Save running results
        os.makedirs("Saved_Results", exist_ok=True)
        pickle_file = os.path.join("Saved_Results", f"{args.variant}_channel_results.pkl")
        with open(pickle_file, "wb") as f:
            pickle.dump(results, f)

    # save best model
    best_model_file = os.path.join("Saved_Results", f"{args.variant}_best_model.pkl")
    with open(best_model_file, "wb") as f:
        pickle.dump(best_model, f)

    # saved averaged confusion matrix across the three iterations for the best model
    np.save(os.path.join("Saved_Results", f"{args.variant}_best_conf_matrix.npy"),
            best_conf_matrix)

    save_confusion_matrix(best_conf_matrix,
                          out_dir="Saved_Results",
                          name=f"{args.variant}_best_conf_matrix")
    

    ranked_results = sorted(results, key=lambda x: x["avg_f1"], reverse=True)

    print(f"\n===== RANKED CHANNEL COMBINATIONS (Best -> Worst) for variant {args.variant} =====")
    for rank, entry in enumerate(ranked_results, 1):
        print(f"{rank}. {entry['combo']}: Avg F1 = {entry['avg_f1']:.4f}, Avg Acc = {entry['avg_acc']:.4f}")

    print("\n===== BEST COMBINATION =====")
    print(f"{best_combo_name} — F1={best_f1:.4f}")
    print(f"Best model saved to: {best_model_file}")


if __name__ == "__main__":
    main()
