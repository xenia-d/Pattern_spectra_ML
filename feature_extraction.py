import os
import glob
import numpy as np
from itertools import combinations
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from lvq.IAALVQ import IAALVQ
from utils.io_management import *
from utils.preprocessing import *
import matplotlib.pyplot as plt


def save_confusion_matrix(conf_matrix, out_dir="Saved_Results", name="confusion_matrix"):
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(6, 5))
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
            plt.text(j, i, f"{conf_matrix[i, j]:.2f}",
                     horizontalalignment="center",
                     color="black" if conf_matrix[i, j] > thresh else "white")

    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()

    save_path = os.path.join(out_dir, f"{name}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")


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


def create_feature_vector(files, channels):
    arrays = []
    for ch in channels:
        arr = np.array(load_granulometry_m_file(files[ch]))[:10, :10].flatten()
        if np.max(arr) == 0:
            return None
        arrays.append(arr)
    return np.concatenate(arrays, axis=0)


def random_splitter_gen(imgs, labels, validation_fraction):
    state = np.random.get_state()

    while True:
        global_state = np.random.get_state()
        np.random.set_state(state)

        I = np.random.permutation(len(imgs))
        n_val = int(round(len(imgs) * validation_fraction))

        imgs_val = [imgs[i] for i in I[:n_val]]
        labels_val = labels[I[:n_val]]

        imgs_train = [imgs[i] for i in I[n_val:]]
        labels_train = labels[I[n_val:]]

        state = np.random.get_state()
        np.random.set_state(global_state)

        yield imgs_train, labels_train, imgs_val, labels_val


def load_channel_files(root_dir, label_prefix):
    channel_files = {}
    for ch in ["R", "G", "H", "V"]:
        pattern = os.path.join(root_dir, f"{label_prefix}{ch}channel", f"{label_prefix}{ch}*.m")
        files = sorted(glob.glob(pattern))
        channel_files[ch] = files
    return channel_files


def main():
    ROOT_DIR = "xmaxtree/output/Spunta"
    xval_count = 5

    label_map = {"Healthy": "h", "Unhealthy": "uh"}
    results = []

    # Generate all single, pairwise, and 3-channel combinations
    channel_pool = ["R", "G", "H", "V"]
    all_combinations = []
    for r in range(1, 4):
        all_combinations.extend(combinations(channel_pool, r))

    for combo in all_combinations:
        print(f"\n=== Running experiment for channels: {combo} ===")
        all_features = []
        all_labels = []

        for label_name, prefix in label_map.items():
            files = load_channel_files(ROOT_DIR, prefix)
            feats = []

            n_samples = len(files["R"])  # assume all channels have same number of files
            for i in range(n_samples):
                file_dict = {ch: files[ch][i] for ch in combo}
                feat = create_feature_vector(file_dict, combo)
                if feat is not None:
                    feats.append(feat)

            feats = np.array(feats)
            labels = np.array([0 if label_name == "Healthy" else 1] * len(feats))

            all_features.append(feats)
            all_labels.append(labels)

        X = np.vstack(all_features)
        y = np.concatenate(all_labels)

        splitter = random_splitter_gen(X, y, 0.2)
        all_f1_test = []
        all_acc_test = []
        all_conf = []

        for fold in range(xval_count):
            x_train, y_train, x_test, y_test = next(splitter)

            model = IAALVQ(max_iter=100, prototypes_per_class=2, omega_rank=400,
                            seed=59, regularization=0.00001, omega_locality='PW',
                            filter_bank=None, block_eye=False, norm=False,
                            correct_imbalance=True)

            x_train = np.array(x_train)
            x_test = np.array(x_test)
            y_train = np.array(y_train, dtype=np.int64)
            y_test = np.array(y_test, dtype=np.int64)

            model.fit(x_train, y_train)
            y_pred_test = model.predict(x_test)

            f1 = f1_score(y_test, y_pred_test, average='weighted')
            acc = accuracy_score(y_test, y_pred_test)
            conf = confusion_matrix(y_test, y_pred_test, normalize='true')

            all_f1_test.append(f1)
            all_acc_test.append(acc)
            all_conf.append(conf)

        avg_f1 = np.mean(all_f1_test)
        avg_acc = np.mean(all_acc_test)
        avg_conf = np.mean(all_conf, axis=0)

        combo_name = "_".join(combo)
        save_confusion_matrix(avg_conf, out_dir="Saved_Results", name=f"conf_{combo_name}")
        np.save(os.path.join("Saved_Results", f"f1_{combo_name}.npy"), np.array(all_f1_test))
        np.save(os.path.join("Saved_Results", f"acc_{combo_name}.npy"), np.array(all_acc_test))

        results.append((combo_name, avg_f1, avg_acc))
        print(f"Channels {combo}: Avg weighted F1 = {avg_f1:.4f}, Avg Accuracy = {avg_acc:.4f}")

    # Rank combinations by F1
    ranked_results = sorted(results, key=lambda x: x[1], reverse=True)
    print("\n===== RANKED CHANNEL COMBINATIONS (Best -> Worst) =====")
    for rank, (combo_name, f1, acc) in enumerate(ranked_results, 1):
        print(f"{rank}. {combo_name}: F1 = {f1:.4f}, Accuracy = {acc:.4f}")


if __name__ == "__main__":
    main()
