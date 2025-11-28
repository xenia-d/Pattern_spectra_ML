import os
import glob
import numpy as np
from lvq.IAALVQ import IAALVQ
from sklearn.metrics import (
    f1_score, accuracy_score, confusion_matrix,
    precision_recall_fscore_support
)
import argparse


def load_granulometry_m_file(path):
    data = []
    with open(path, 'r') as f:
        lines = f.readlines()[2:]  # skip headers
    for line in lines:
        line = line.strip()
        if not line:
            continue
        numbers = [float(x) for x in line.split()[1:]]
        data.append(numbers)
    return np.array(data)


def create_feature_vector(file_dict, channels, dropped_counter):
    arrays = []
    for ch in channels:
        arr = np.array(load_granulometry_m_file(file_dict[ch]))[:10, :10].flatten()
        if np.max(arr) == 0:
            dropped_counter[ch] += 1
            return None
        arrays.append(arr)
    return np.concatenate(arrays, axis=0)


def random_splitter_gen(imgs, labels, validation_fraction):
    imgs = np.array(imgs)
    labels = np.array(labels)
    I = np.random.permutation(len(imgs))
    n_val = max(1, int(round(len(imgs) * validation_fraction)))
    return imgs[I[n_val:]], labels[I[n_val:]], imgs[I[:n_val]], labels[I[:n_val]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, default="Rudolph")
    parser.add_argument("--channels", nargs="+", default=["R","H","V"])
    args = parser.parse_args()
    
    CHANNELS_TO_TEST = tuple(args.channels)
    print("\n===== DIAGNOSTIC RUN =====")
    print("Variant:", args.variant)
    print("Channels:", CHANNELS_TO_TEST)

    ROOT_DIR = f"xmaxtree/output/{args.variant}"
    label_map = {"Healthy": "h", "Unhealthy": "uh"}

    all_features = []
    all_labels = []

    # track dropped samples
    dropped = {"Healthy": {c:0 for c in CHANNELS_TO_TEST},
               "Unhealthy": {c:0 for c in CHANNELS_TO_TEST}}

    # ---- Load features ----
    for label_name, prefix in label_map.items():
        files_dict = {}
        for ch in CHANNELS_TO_TEST:
            files_dict[ch] = sorted(
                glob.glob(os.path.join(ROOT_DIR,
                                       f"{label_name}Leaf_{ch}channel",
                                       f"{prefix}{ch}*.m"))
            )
        n_samples = len(files_dict[CHANNELS_TO_TEST[0]])
        feats = []
        for i in range(n_samples):
            file_dict = {ch: files_dict[ch][i] for ch in CHANNELS_TO_TEST}
            feat = create_feature_vector(file_dict, CHANNELS_TO_TEST, dropped[label_name])
            if feat is not None:
                feats.append(feat)

        feats = np.array(feats)
        labels = np.array([0 if label_name == "Healthy" else 1] * len(feats))
        all_features.append(feats)
        all_labels.append(labels)

    X = np.vstack(all_features)
    y = np.concatenate(all_labels)

    # ---- Print class distribution & dropped samples ----
    uniq, counts = np.unique(y, return_counts=True)
    print("\nClass distribution BEFORE split:", dict(zip(uniq, counts)))
    print("\nDropped feature vectors:")
    print("Healthy:", dropped["Healthy"])
    print("Unhealthy:", dropped["Unhealthy"])

    # ---- Single split ----
    x_train, y_train, x_test, y_test = random_splitter_gen(X, y, validation_fraction=0.2)

    print("\nTrain distribution:", dict(zip(*np.unique(y_train, return_counts=True))))
    print("Test distribution:", dict(zip(*np.unique(y_test, return_counts=True))))

    # ---- Train model ----
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

    # ---- Metrics ----
    print("\n===== METRICS =====")

    # raw confusion
    conf_raw = confusion_matrix(y_test, y_pred)
    print("\nRaw confusion matrix:\n", conf_raw)

    conf_norm = confusion_matrix(y_test, y_pred, normalize='true')
    print("\nNormalized confusion matrix:\n", conf_norm)

    # per-class metrics
    p, r, f, s = precision_recall_fscore_support(
        y_test, y_pred, labels=[0,1], zero_division=0
    )

    print(f"\nPer-class metrics:")
    print(f"Healthy   P={p[0]:.3f}, R={r[0]:.3f}, F1={f[0]:.3f}, Support={s[0]}")
    print(f"Unhealthy P={p[1]:.3f}, R={r[1]:.3f}, F1={f[1]:.3f}, Support={s[1]}")

    macro_f1 = f.mean()
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"\nMacro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"Accuracy:    {accuracy_score(y_test, y_pred):.4f}")

    print("\n===== END DIAGNOSTIC =====")


if __name__ == "__main__":
    main()
