# RG-channel classifier version of the original script
# (Simplified to only load R and G channels, same pipeline)

import os
import glob
import numpy as np
import random
from sklearn.metrics import f1_score, confusion_matrix
from lvq.IAALVQ import IAALVQ
from utils.io_management import *
from utils.preprocessing import *
import matplotlib.pyplot as plt


def save_confusion_matrix(conf_matrix, out_dir="Saved_Results"):
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(6, 5))
    plt.imshow(conf_matrix, interpolation="nearest")
    plt.title("Confusion Matrix (Normalized)")
    plt.colorbar()

    classes = ["Healthy", "Unhealthy"]

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Print the numbers inside the matrix
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, f"{conf_matrix[i, j]:.2f}",
                     horizontalalignment="center",
                     color="black" if conf_matrix[i, j] > thresh else "white")

    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()

    save_path = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")



def load_granulometry_m_file(path):
    data = []
    with open(path, 'r') as f:
        lines = f.readlines()
    lines = lines[2:]
    for line in lines:
        line = line.strip()
        if not line:
            continue
        tokens = line.split()
        numbers = [float(x) for x in tokens[1:]]
        data.append(numbers)
    return np.array(data)


def create_rg_feature_vector(R_file, G_file):
    R = np.array(load_granulometry_m_file(R_file))[:10, :10].flatten()
    if np.max(R) == 0:
        return None

    G = np.array(load_granulometry_m_file(G_file))[:10, :10].flatten()
    if np.max(G) == 0:
        return None

    if len(R) != len(G):
        raise ValueError("R and G arrays must be the same length")

    return np.concatenate([R, G], axis=0)


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


def main():
    ROOT_DIR = "xmaxtree/output/Spunta"

    # Healthy leaf RG
    R_files = sorted(glob.glob(os.path.join(ROOT_DIR, "HealthyLeaf_Rchannel", "hR*.m")))
    G_files = sorted(glob.glob(os.path.join(ROOT_DIR, "HealthyLeaf_Gchannel", "hG*.m")))

    all_histogramsH = []
    for R_file, G_file in zip(R_files, G_files):
        feat = create_rg_feature_vector(R_file, G_file)
        if feat is not None:
            all_histogramsH.append(feat)
    all_histogramsH = np.array(all_histogramsH)

    # Unhealthy leaf RG
    R_files = sorted(glob.glob(os.path.join(ROOT_DIR, "UnhealthyLeaf_Rchannel", "uhR*.m")))
    G_files = sorted(glob.glob(os.path.join(ROOT_DIR, "UnhealthyLeaf_Gchannel", "uhG*.m")))

    all_histogramsNH = []
    for R_file, G_file in zip(R_files, G_files):
        feat = create_rg_feature_vector(R_file, G_file)
        if feat is not None:
            all_histogramsNH.append(feat)
    all_histogramsNH = np.array(all_histogramsNH)

    # Stack into dataset
    X = np.vstack([all_histogramsH, all_histogramsNH])
    y = np.array([0] * len(all_histogramsH) + [1] * len(all_histogramsNH))

    xval_count = 5
    splitter = random_splitter_gen(X, y, 0.2)

    all_f1_test = []
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
        conf = confusion_matrix(y_test, y_pred_test, normalize='true')

        print(f"Fold {fold+1} Test F1: {f1}")
        print(conf)

        all_f1_test.append(f1)
        all_conf.append(conf)

    print("\n========== FINAL RESULTS (RG ONLY) ==========")
    print("Final weighted Test F1-score:", np.mean(all_f1_test))
    print("Final averaged confusion matrix:\n", np.mean(all_conf, axis=0))
    print("============================================\n")

    save_confusion_matrix(np.mean(all_conf, axis=0), out_dir="Saved_Results")

if __name__ == "__main__":
    main()
