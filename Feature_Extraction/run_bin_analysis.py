import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)


import zipfile
import glob
import argparse
import pickle
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from lvq.IAALVQ import IAALVQ
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import torch

np.random.seed(12)
torch.manual_seed(12)

DEFAULT_OUT_DIR = "Saved_Results"
CHANNEL_POOL = ["R", "G", "B", "H", "S", "V"]
LABEL_MAP = {"Healthy": "h", "Unhealthy": "uh"}
DEFAULT_XVAL_COUNT = 5
DEFAULT_XVAL_FRACTION = 0.2
DEFAULT_ITERATIONS = 3

# def visualize_mask(arr_10x10, shape_bins, size_bins, mode="insertion"):

#     if mode == "insertion":
#         # selection mask
#         mask = np.zeros_like(arr_10x10, dtype=bool)
#         mask[np.ix_(shape_bins, size_bins)] = True
#     else:  # deletion
#         mask = np.ones_like(arr_10x10, dtype=bool)
#         mask[np.ix_(shape_bins, size_bins)] = False

#     fig, ax = plt.subplots(1, 2, figsize=(10, 4))

#     ax[0].imshow(arr_10x10, cmap="viridis")
#     ax[0].set_title("Original 10×10 Pattern Spectrum")

#     overlay = np.ma.masked_where(~mask, arr_10x10)
#     ax[1].imshow(arr_10x10, cmap="gray", alpha=0.3)
#     ax[1].imshow(overlay, cmap="coolwarm", alpha=0.9)
#     ax[1].set_title(f"Selected Bins ({mode})")

#     plt.show()



def make_model():
    return IAALVQ(
        max_iter=100,
        prototypes_per_class=2,
        omega_rank=400,
        seed=59,
        regularization=1e-5,
        omega_locality="PW",
        filter_bank=None,
        block_eye=False,
        norm=False,
        correct_imbalance=True,
    )


def save_confusion_matrix(conf_matrix, out_dir=DEFAULT_OUT_DIR, name="confusion_matrix"):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(6, 5))
    plt.imshow(conf_matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (Normalized)")
    plt.colorbar()
    classes = ["Healthy", "Unhealthy"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = conf_matrix.max() / 2.0 if conf_matrix.size else 0.5
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(
                j,
                i,
                f"{conf_matrix[i, j]:.2f}",
                horizontalalignment="center",
                color="white" if conf_matrix[i, j] > thresh else "black",
            )
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    path = os.path.join(out_dir, f"{name}.png")
    plt.savefig(path, dpi=300)
    plt.close()
    np.save(os.path.join(out_dir, f"{name}.npy"), conf_matrix)
    print(f"Saved confusion matrix to {path} and .npy")


def load_granulometry_m_file(path):
    data = []
    with open(path, "r") as f:
        lines = f.readlines()[2:]  # skip headers
    for line in lines:
        tokens = line.strip().split()
        if tokens:
            data.append([float(x) for x in tokens[1:]])
    return np.array(data)


def create_feature_array_from_10x10(arr_10x10, shape_bins, size_bins): # INSERTION CASE
    sel = arr_10x10[np.ix_(shape_bins, size_bins)]
    if np.max(sel) == 0:
        return None
    return sel.flatten()

def create_feature_array_deletion(arr_10x10, shape_bins_to_remove, size_bins_to_remove):
    # return all bins except those in shape_bins_to_remove and size_bins_to_remove - FOR DELETION CASE
    mask = np.ones((11, 11), dtype=bool)
    mask[np.ix_(shape_bins_to_remove, size_bins_to_remove)] = False
    sel = arr_10x10[mask] # apply removal mask 

    if np.max(sel) == 0:
        return None
    
    return sel

def random_splitter_gen(imgs, labels, validation_fraction):
    imgs = np.array(imgs)
    labels = np.array(labels)
    while True:
        I = np.random.permutation(len(imgs))
        n_val = max(1, int(round(len(imgs) * validation_fraction)))
        yield imgs[I[n_val:]], labels[I[n_val:]], imgs[I[:n_val]], labels[I[:n_val]]


def evaluate_fold(x_train, y_train, x_test, y_test):
    model = make_model()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    f1 = f1_score(y_test, y_pred, average="weighted")
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    conf = confusion_matrix(y_test, y_pred, normalize="true")
    return f1, acc, prec, rec, conf


def run_cv_iterations(X, y, xval_count, xval_fraction, iterations):
    iter_results = []
    confs_accum = []
    for seed in range(iterations):
        np.random.seed(seed)
        splitter = random_splitter_gen(X, y, xval_fraction)
        f1s, accs, precs, recs = [], [], [], []
        confs = []
        for _fold in range(xval_count):
            x_train, y_train, x_val, y_val = next(splitter)
            f1, acc, prec, rec, conf = evaluate_fold(x_train, y_train, x_val, y_val)
            f1s.append(f1)
            accs.append(acc)
            precs.append(prec)
            recs.append(rec)
            confs.append(conf)
        iter_results.append(
            {
                "seed": seed,
                "f1_mean": float(np.mean(f1s)),
                "acc_mean": float(np.mean(accs)),
                "prec_mean": float(np.mean(precs)),
                "rec_mean": float(np.mean(recs)),
                "conf_mean": np.mean(confs, axis=0).tolist(),
            }
        )
        confs_accum.append(np.mean(confs, axis=0))
    if confs_accum:
        avg_conf = np.mean(confs_accum, axis=0)
    else:
        avg_conf = np.zeros((2, 2))
    return iter_results, avg_conf


def build_train_test_indices(n_samples, xval_fraction):
    # 80-20 split, randomly select 20% of indices for test, rest for train
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    n_test = max(1, int(round(n_samples * xval_fraction)))
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return train_idx, test_idx


def collect_preloaded(ROOT_DIR, label_map=LABEL_MAP, channels_to_load=None):
    preloaded = {}
    if channels_to_load is None:
        channels_to_load = CHANNEL_POOL
    for label_name, prefix in label_map.items():
        preloaded[label_name] = {}
        for ch in channels_to_load:
            pattern_files = sorted(
                glob.glob(os.path.join(ROOT_DIR, f"{label_name}Leaf_{ch}channel", f"{prefix}{ch}*.m"))
            )
            arrs = []
            for p in pattern_files:
                arr = load_granulometry_m_file(p)
                if arr.size == 0:
                    continue
                else:
                    arr = np.array(arr)
                arrs.append(np.array(arr, dtype=float))
            preloaded[label_name][ch] = np.array(arrs)
    return preloaded

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, required=True, help="Variant folder (Spunta, Mondial, Fontane, Rudolph)")
    parser.add_argument("--combo", type=str, default=None, help="specify best combo as underscore-separated channels, e.g. 'R_B' or 'R_G'")
    parser.add_argument("--eval_type", type=str, default="insertion", choices=["insertion", "deletion"], help="Choose whether to evaluate using insertion or deletion of bin groups")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUT_DIR, help="Where to save results")
    args = parser.parse_args()

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
    VARIANT = args.variant
    ROOT_DIR = os.path.join(PROJECT_ROOT, "xmaxtree", "output", VARIANT)
    OUT_DIR = args.output_dir
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Running bin analysis for variant {args.variant} with {args.eval_type}")

    # unzip fontane data
    if VARIANT == "Fontane":
        fontane_zip = os.path.join(PROJECT_ROOT, "xmaxtree", "output", "Fontane.zip")
        if os.path.exists(fontane_zip):
            with zipfile.ZipFile(fontane_zip, 'r') as zip_ref:
                zip_ref.extractall(os.path.join(PROJECT_ROOT, "xmaxtree", "output"))
            print("Extracted Fontane.zip")

    # because fontane has to big of a dataset - too much computation for 5 folds 
    if args.variant == "Fontane":
        xval_count = 3
    else :
        xval_count = DEFAULT_XVAL_COUNT    

    xval_fraction = DEFAULT_XVAL_FRACTION
    iterations = DEFAULT_ITERATIONS

    print("determining combination")

    #  Determine combo first -
    if args.combo and args.combo.strip():
        best_combo = tuple(args.combo.strip().split("_"))
        print(f"Using provided combo: {best_combo}")

    #  Preload only needed channels 
    print(f"Preloading pattern spectra for channels: {best_combo} ...")
    preloaded = collect_preloaded(ROOT_DIR, LABEL_MAP, channels_to_load=best_combo)
    print("Preload complete.")


    # build train/test indices
    label_train_test_indices = {}
    for label_name in LABEL_MAP:
        n_samples_label = preloaded[label_name][CHANNEL_POOL[0]].shape[0]
        train_idx, test_idx = build_train_test_indices(n_samples_label, xval_fraction)
        label_train_test_indices[label_name] = (train_idx, test_idx)

    # print label counts per split
    for label_name in LABEL_MAP:
        train_idx, test_idx = label_train_test_indices[label_name]
        print(f" Label '{label_name}' -> Train: {len(train_idx)}, Test: {len(test_idx)}")


    for label_name in LABEL_MAP:
        train_idx, test_idx = label_train_test_indices[label_name]
        for ch in best_combo:
            arrs = preloaded[label_name][ch]
            preloaded[label_name][ch + "_train"] = arrs[train_idx]
            preloaded[label_name][ch + "_test"] = arrs[test_idx]


    #  ----- Define bin groups to evaluate -----

    # Include the 10th bin now (0..10)
    elongated_shape_bins = list(range(0, 6))  # 0..5
    compact_shape_bins = list(range(6, 11)) # 6..10
    size_bins = list(range(11))   # 0..10

    small_size_bins = list(range(0, 6))   # 0..5
    large_size_bins = list(range(6, 11))  # 6..10

    block_names = ["elongated_small", "elongated_large", "compact_small", "compact_large"]

    subset_specs = []

    # small sizes (all shapes)
    subset_specs.append({
        "kind": "small_sizes",
        "shape_bins": elongated_shape_bins + compact_shape_bins,
        "size_bins": small_size_bins
    })

    # large sizes (all shapes)
    subset_specs.append({
        "kind": "large_sizes",
        "shape_bins": elongated_shape_bins + compact_shape_bins,
        "size_bins": large_size_bins
    })

    # elongated shapes (all sizes)
    subset_specs.append({
        "kind": "elongated_shapes",
        "shape_bins": elongated_shape_bins,
        "size_bins": small_size_bins + large_size_bins
    })
    # compact shapes (all sizes)
    subset_specs.append({
        "kind": "compact_shapes",
        "shape_bins": compact_shape_bins,
        "size_bins": small_size_bins + large_size_bins
    })

    
    grouped_blocks = [
        (elongated_shape_bins, small_size_bins), # elongated + small
        (elongated_shape_bins, large_size_bins), # elongated + large
        (compact_shape_bins, small_size_bins),   # compact + small
        (compact_shape_bins, large_size_bins),   # compact + large
    ]

    # blocks - combined shape and size
    for name, (s_bins, z_bins) in zip(block_names, grouped_blocks):
        subset_specs.append({
            "kind": name,
            "shape_bins": s_bins,
            "size_bins": z_bins
        })

    print(f"Will evaluate {len(subset_specs)} subsets per channel. Subset kinds: {[s['kind'] for s in subset_specs]}")



    # Per-channel bin analysis

    for ch in best_combo:
        print("\n" + "="*60)
        print(f"Running bin-analysis for variant={VARIANT}, channel={ch}")

        # gather train/test arrays across labels for this channel
        X_train_per_label = []
        y_train_per_label = []
        X_test_per_label = []
        y_test_per_label = []
        for label_idx, label_name in enumerate(LABEL_MAP.keys()):
            Xtr_arrs = preloaded[label_name][ch + "_train"]
            Xte_arrs = preloaded[label_name][ch + "_test"]
            ntr = Xtr_arrs.shape[0]
            nte = Xte_arrs.shape[0]
            X_train_per_label.append(Xtr_arrs)
            y_train_per_label.append(np.array([label_idx] * ntr))
            X_test_per_label.append(Xte_arrs)
            y_test_per_label.append(np.array([label_idx] * nte))

        X_train_full_10x10 = np.vstack(X_train_per_label)
        y_train_full = np.concatenate(y_train_per_label)
        X_test_full_10x10 = np.vstack(X_test_per_label)
        y_test_full = np.concatenate(y_test_per_label)

        channel_results = []

        for spec in subset_specs:
            kind = spec["kind"]
            s_bins = spec["shape_bins"]
            z_bins = spec["size_bins"]
            print(f"\n-> Subset: {kind} | shapes {s_bins} | sizes {z_bins}")

            # visualize_mask(X_train_full_10x10[0], s_bins, z_bins, mode=args.eval_type)

            # Build feature vectors (skip samples where block is all-zero)
            Xtr_list = []
            ytr_list = []
            for xi, yi in zip(X_train_full_10x10, y_train_full):
                if args.eval_type == "insertion":
                    fv = create_feature_array_from_10x10(xi, s_bins, z_bins)
                else:  # deletion
                    fv = create_feature_array_deletion(xi, s_bins, z_bins)
                if fv is None:
                    print("  skipping train sample due to all-zero feature vector")
                    continue
                Xtr_list.append(fv)
                ytr_list.append(yi)
            Xtr_list = np.array(Xtr_list)
            ytr_list = np.array(ytr_list, dtype=np.int64)

            Xte_list = []
            yte_list = []
            for xi, yi in zip(X_test_full_10x10, y_test_full):
                if args.eval_type == "insertion":
                    fv = create_feature_array_from_10x10(xi, s_bins, z_bins)
                else:  # deletion
                    fv = create_feature_array_deletion(xi, s_bins, z_bins)
                if fv is None:
                    continue
                Xte_list.append(fv)
                yte_list.append(yi)
            Xte_list = np.array(Xte_list)
            yte_list = np.array(yte_list, dtype=np.int64)

            iter_metrics, avg_conf = run_cv_iterations(Xtr_list, ytr_list, xval_count, xval_fraction, iterations)

            # build entry for saving
            f1_by_iter = np.array([it["f1_mean"] for it in iter_metrics])
            acc_by_iter = np.array([it["acc_mean"] for it in iter_metrics])
            prec_by_iter = np.array([it["prec_mean"] for it in iter_metrics])
            rec_by_iter = np.array([it["rec_mean"] for it in iter_metrics])

            entry = {
                "kind": kind,
                "shape_bins": s_bins,
                "size_bins": z_bins,
                "n_train_after_filter": int(Xtr_list.shape[0]),
                "n_test_after_filter": int(Xte_list.shape[0]),
                "iter_metrics": iter_metrics,
                "f1_mean_across_iters": float(f1_by_iter.mean()),
                "f1_std_across_iters": float(f1_by_iter.std(ddof=0)),
                "acc_mean_across_iters": float(acc_by_iter.mean()),
                "acc_std_across_iters": float(acc_by_iter.std(ddof=0)),
                "prec_mean_across_iters": float(prec_by_iter.mean()),
                "prec_std_across_iters": float(prec_by_iter.std(ddof=0)),
                "rec_mean_across_iters": float(rec_by_iter.mean()),
                "rec_std_across_iters": float(rec_by_iter.std(ddof=0)),
                "avg_conf": avg_conf.tolist(),
            }

            channel_results.append(entry)
            print(f"   -> {kind}: avg F1 across iters = {entry['f1_mean_across_iters']:.4f} (std {entry['f1_std_across_iters']:.4f})")

        # Save per-channel results
        out_pkl = os.path.join(OUT_DIR, f"{VARIANT}_bin_analysis_{ch}.pkl")
        with open(out_pkl, "wb") as f:
            pickle.dump(channel_results, f)
        print(f"\nSaved bin-analysis results for channel {ch} to {out_pkl}")

        # Pick best subset for this channel
        valid_results = [r for r in channel_results if "f1_mean_across_iters" in r]
        best_subset = max(valid_results, key=lambda rr: rr["f1_mean_across_iters"])
        print(f"\nBest subset for channel {ch}: {best_subset['kind']} (avg F1={best_subset['f1_mean_across_iters']:.4f})")

        # --------- Retrain on entire train set using best subset and evaluate on held-out test set using best bin combos -----------
        s_bins = best_subset["shape_bins"]
        z_bins = best_subset["size_bins"]

        Xtr_final_list = []
        ytr_final_list = []
        for xi, yi in zip(X_train_full_10x10, y_train_full):
            if args.eval_type == "insertion":
                fv = create_feature_array_from_10x10(xi, s_bins, z_bins)
            else:
                fv = create_feature_array_deletion(xi, s_bins, z_bins)
            if fv is None:
                print("  skipping train sample due to all-zero feature vector")
                continue
            Xtr_final_list.append(fv)
            ytr_final_list.append(yi)
        Xtr_final = np.array(Xtr_final_list)
        ytr_final = np.array(ytr_final_list, dtype=np.int64)

        Xte_final_list = []
        yte_final_list = []
        for xi, yi in zip(X_test_full_10x10, y_test_full):
            if args.eval_type == "insertion":
                fv = create_feature_array_from_10x10(xi, s_bins, z_bins)
            else:
                fv = create_feature_array_deletion(xi, s_bins, z_bins)
            if fv is None:
                print("  skipping test sample due to all-zero feature vector")
                continue
            Xte_final_list.append(fv)
            yte_final_list.append(yi)
        Xte_final = np.array(Xte_final_list)
        yte_final = np.array(yte_final_list, dtype=np.int64)


        print("\nFINAL TRAIN/TEST AFTER BIN FILTER:")
        print("Train Healthy:   ", np.sum(np.array(ytr_final) == 0))
        print("Train Unhealthy: ", np.sum(np.array(ytr_final) == 1))
        print("Test Healthy:    ", np.sum(np.array(yte_final) == 0))
        print("Test Unhealthy:  ", np.sum(np.array(yte_final) == 1))
        print("Total Train:", len(ytr_final), "Total Test:", len(yte_final))


        # train final per-channel model and evaluate on test set
        final_model = make_model()
        final_model.fit(Xtr_final, ytr_final)
        y_pred_test = final_model.predict(Xte_final)

        final_f1 = f1_score(yte_final, y_pred_test, average="weighted")
        final_acc = accuracy_score(yte_final, y_pred_test)
        final_prec_w = precision_score(yte_final, y_pred_test, average="weighted", zero_division=0)
        final_prec_m = precision_score(yte_final, y_pred_test, average="macro", zero_division=0)
        final_rec_w = recall_score(yte_final, y_pred_test, average="weighted", zero_division=0)
        final_rec_m = recall_score(yte_final, y_pred_test, average="macro", zero_division=0)
        final_conf = confusion_matrix(yte_final, y_pred_test, normalize="true")

        print("\n===== FINAL TEST SET RESULTS (best subset) =====")
        print(f"Channel {ch} | Best subset {best_subset['kind']}")
        print(f"Test Weighted F1: {final_f1:.4f}")
        print(f"Test Accuracy   : {final_acc:.4f}")
        print(f"Test Precision Weighted: {final_prec_w:.4f}")
        print(f"Test Precision Macro  : {final_prec_m:.4f}")
        print(f"Test Recall Weighted  : {final_rec_w:.4f}")
        print(f"Test Recall Macro     : {final_rec_m:.4f}")
        print("Test Confusion Matrix (normalized):")
        print(final_conf)

        # save model + confusion matrix + metadata
        model_path = os.path.join(OUT_DIR, f"{VARIANT}_{ch}_best_subset_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(final_model, f)
        save_confusion_matrix(final_conf, out_dir=OUT_DIR, name=f"{VARIANT}_{ch}_best_subset_conf_matrix")
        meta = {
            "variant": VARIANT,
            "channel": ch,
            "best_subset": best_subset,
            "final_metrics": {"f1": float(final_f1), "acc": float(final_acc), "prec_w": float(final_prec_w), "prec_m": float(final_prec_m), "rec_w": float(final_rec_w), "rec_m": float(final_rec_m)},
            "model_path": model_path,
        }
        with open(os.path.join(OUT_DIR, f"{VARIANT}_{ch}_best_subset_meta.pkl"), "wb") as f:
            pickle.dump(meta, f)
        print(f"Saved final model to {model_path} and metadata.")

    print("\nAll channels processed. Bin-analysis complete.")

    print("\n-------------------------------------------------------------")
    print("FINAL MULTI-CHANNEL EVALUATION WITH COMBINED BEST SUBSETS")
    print("----------------------------------------------------------------")

    # Load best subsets per channel
    all_best = {}
    for ch in best_combo:
        pkl_path = os.path.join(OUT_DIR, f"{VARIANT}_bin_analysis_{ch}.pkl")
        results = pickle.load(open(pkl_path, "rb"))
        valid = [r for r in results if "f1_mean_across_iters" in r]
        best = max(valid, key=lambda rr: rr["f1_mean_across_iters"])
        all_best[ch] = best


    print("Best subsets per channel:")
    for ch, b in all_best.items():
        print(f"  {ch}: {b['kind']}  (shapes={b['shape_bins']}, sizes={b['size_bins']})")

    # Build combined features (concatenate per-channel best-bin features)

    Xtr_final = []
    ytr_final = []
    Xte_final = []
    yte_final = []


    for label_idx, label_name in enumerate(LABEL_MAP.keys()):

        # Number of train samples for this label
        n_train = len(X_train_per_label[label_idx])

        for idx in range(n_train):
            fv_parts = []
            skip = False

            for ch in best_combo:
                xi = X_train_per_label[label_idx][idx]

                s_bins = all_best[ch]["shape_bins"]
                z_bins = all_best[ch]["size_bins"]

                if args.eval_type == "insertion":
                    fv = create_feature_array_from_10x10(xi, s_bins, z_bins)
                else:
                    fv = create_feature_array_deletion(xi, s_bins, z_bins)

                if fv is None or fv.size == 0:
                    skip = True
                    break

                fv_parts.append(fv)

            if not skip:
                Xtr_final.append(np.concatenate(fv_parts))
                ytr_final.append(label_idx)

    # BUILD FINAL TEST FEATURES
    for label_idx, label_name in enumerate(LABEL_MAP.keys()):

        n_test = len(X_test_per_label[label_idx])

        for idx in range(n_test):
            fv_parts = []
            skip = False

            for ch in best_combo:
                xi = X_test_per_label[label_idx][idx]

                s_bins = all_best[ch]["shape_bins"]
                z_bins = all_best[ch]["size_bins"]

                if args.eval_type == "insertion":
                    fv = create_feature_array_from_10x10(xi, s_bins, z_bins)
                else:
                    fv = create_feature_array_deletion(xi, s_bins, z_bins)

                if fv is None or fv.size == 0:
                    skip = True
                    break

                fv_parts.append(fv)

            if not skip:
                Xte_final.append(np.concatenate(fv_parts))
                yte_final.append(label_idx)

    Xtr_final = np.array(Xtr_final)
    ytr_final = np.array(ytr_final, dtype=np.int64)
    Xte_final = np.array(Xte_final)
    yte_final = np.array(yte_final, dtype=np.int64)


    print(f"\nTraining final multi-channel model using {len(best_combo)} channels...")

    # train on full train set and evaluate on test set
    multi_model = make_model()
    multi_model.fit(Xtr_final, ytr_final)
    y_pred_final = multi_model.predict(Xte_final)

    final_f1 = f1_score(yte_final, y_pred_final, average="weighted")
    final_acc = accuracy_score(yte_final, y_pred_final)
    final_prec_w = precision_score(yte_final, y_pred_final, average="weighted", zero_division=0)
    final_prec_m = precision_score(yte_final, y_pred_final, average="macro", zero_division=0)
    final_rec_w = recall_score(yte_final, y_pred_final, average="weighted", zero_division=0)
    final_rec_m = recall_score(yte_final, y_pred_final, average="macro", zero_division=0)
    final_conf = confusion_matrix(yte_final, y_pred_final, normalize="true")

    print("\n******** FINAL MULTI-CHANNEL TEST RESULTS ********")
    print(f"Channels used: {best_combo}")
    print(f"Weighted F1  : {final_f1:.4f}")
    print(f"Accuracy     : {final_acc:.4f}")
    print(f"Precision Weighted: {final_prec_w:.4f}")
    print(f"Precision Macro  : {final_prec_m:.4f}")
    print(f"Recall Weighted  : {final_rec_w:.4f}")
    print(f"Recall Macro     : {final_rec_m:.4f}")
    print("Confusion Matrix (normalized):")
    print(final_conf)

    save_confusion_matrix(final_conf, out_dir=OUT_DIR, name=f"{VARIANT}_multi_channel_best_subsets_conf_matrix")

    meta = {
        "variant": VARIANT,
        "channels": best_combo,
        "best_subsets": all_best,
        "final_metrics": {"f1": float(final_f1), "acc": float(final_acc), "prec_w": float(final_prec_w), "prec_m": float(final_prec_m), "rec_w": float(final_rec_w), "rec_m": float(final_rec_m)},
    }
    with open(os.path.join(OUT_DIR, f"{VARIANT}_multi_channel_best_subsets_meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    # Save final trained multi-channel model
    model_path = os.path.join(OUT_DIR, f"{VARIANT}_multi_channel_best_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(multi_model, f)
    print(f"Final multi-channel model saved to: {model_path}")
    print("Saved multi-channel model results.")


if __name__ == "__main__":
    main()
