import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

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
    thresh = conf_matrix.max() / 2.0 if conf_matrix.size else 0.5
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, f"{conf_matrix[i,j]:.2f}",
                     horizontalalignment="center",
                     color="white" if conf_matrix[i,j] > thresh else "black")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    path = os.path.join(out_dir, f"{name}.png")
    plt.savefig(path, dpi=300)
    plt.close()
    # also save numpy
    np.save(os.path.join(out_dir, f"{name}.npy"), conf_matrix)
    print(f"Saved confusion matrix to {path} and .npy")

def load_granulometry_m_file(path):
    data = []
    with open(path, 'r') as f:
        lines = f.readlines()[2:]  # skip headers
    for line in lines:
        tokens = line.strip().split()
        if tokens:
            data.append([float(x) for x in tokens[1:]])
    return np.array(data)

def create_feature_array_from_10x10(arr_10x10, shape_bins, size_bins):
    sel = arr_10x10[np.ix_(shape_bins, size_bins)]
    if np.max(sel) == 0:
        return None
    return sel.flatten()

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
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    conf = confusion_matrix(y_test, y_pred, normalize='true')
    return f1, acc, prec, rec, conf

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, required=True,
                        help="Variant folder (Spunta, Mondial, Fontane, Rudolph, Rudolph, ...)")
    parser.add_argument("--combo", type=str, default="R_G_B",
                        help="Optional: specify best combo as underscore-separated channels, e.g. 'R_B' or 'R_G'. If omitted the script will attempt to read Saved_Results/{variant}_channel_results.pkl and pick top.")
    parser.add_argument("--output_dir", type=str, default="Saved_Results",
                        help="Where to save results")
    args = parser.parse_args()

    VARIANT = args.variant
    ROOT_DIR = os.path.join(PROJECT_ROOT, "xmaxtree", "output", VARIANT)
    OUT_DIR = args.output_dir
    os.makedirs(OUT_DIR, exist_ok=True)

    xval_count = 5
    xval_fraction = 0.2 
    iterations = 3

    label_map = {"Healthy": "h", "Unhealthy": "uh"}
    channel_pool = ["R", "G", "B", "H", "S", "V"]

    print("Preloading 10x10 pattern spectra for all labels & channels (skipping zero vectors per file/channel)...")
    preloaded = {}
    for label_name, prefix in label_map.items():
        preloaded[label_name] = {}
        for ch in channel_pool:
            pattern_files = sorted(glob.glob(os.path.join(ROOT_DIR, f"{label_name}Leaf_{ch}channel", f"{prefix}{ch}*.m")))
            arrs = []
            for p in pattern_files:
                arr = load_granulometry_m_file(p)

                if arr.size == 0:
                    arr10 = np.zeros((10,10))
                else:
                    arr = np.array(arr)
                    if arr.shape[0] < 10 or arr.shape[1] < 10:
                        # pad if necessary
                        arr10 = np.zeros((10,10))
                        h = min(10, arr.shape[0]); w = min(10, arr.shape[1])
                        arr10[:h,:w] = arr[:h,:w]
                    else:
                        arr10 = arr[:10,:10]
                arrs.append(np.array(arr10, dtype=float))
            preloaded[label_name][ch] = np.array(arrs)  # shape (n_files,10,10)
    print("Preload complete.")

    if args.combo:
        best_combo = tuple(args.combo.split("_"))
        print(f"Using provided combo: {best_combo}")
    else:
        #  if the user does not input a combination, auto-load previous channel results to get best combo
        prev_file = os.path.join(OUT_DIR, f"{VARIANT}_channel_results.pkl")
        if os.path.exists(prev_file):
                prev = pickle.load(open(prev_file, "rb"))
                # find the best by average f1
                best_combo = None
                best_f1 = -1
                if isinstance(prev, list) and len(prev):
                    for entry in prev:
                        if isinstance(entry, dict):
                            combo_name = entry.get("combo")
                            avg_f1 = entry.get("avg_f1", -1)
                        elif isinstance(entry, (tuple, list)) and len(entry) >= 2:
                            combo_name = entry[0]
                            avg_f1 = entry[1]
                        else:
                            continue
                        if avg_f1 > best_f1:
                            best_f1 = avg_f1
                            best_combo = tuple(combo_name.split("_"))
                if best_combo is None:
                    raise RuntimeError("Couldn't infer best combo from previous results file.")
                print(f"Auto-loaded best combo from {prev_file}: {best_combo} (avg_f1={best_f1:.4f})")


    # Use number of samples from first channel in pool 
    for label_name in label_map:
        n_samples = preloaded[label_name][channel_pool[0]].shape[0]
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        n_test = max(1, int(round(n_samples * xval_fraction)))
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]
        for ch in channel_pool:
            feats = preloaded[label_name][ch]
            # if channel has fewer files, raise informative error
            if feats.shape[0] != n_samples:
                raise RuntimeError(f"Channel {ch} for label {label_name} has {feats.shape[0]} files but expected {n_samples} (consistent with earlier script assumptions).")
            preloaded[label_name][ch + "_train_idx"] = train_idx
            preloaded[label_name][ch + "_test_idx"] = test_idx

    # store per-label per-channel train/test arrays shaped (n_train, 10,10)
    for label_name in label_map:
        for ch in channel_pool:
            arrs = preloaded[label_name][ch]
            train_idx = preloaded[label_name][ch + "_train_idx"]
            test_idx = preloaded[label_name][ch + "_test_idx"]
            preloaded[label_name][ch + "_train"] = arrs[train_idx]
            preloaded[label_name][ch + "_test"] = arrs[test_idx]

    # Define bin groups to analyze for shape, size and both


    # Shape bins grouped by compactness (moment of inertia)
    less_compact_shape_bins = list(range(0, 5))  #  elongated
    more_compact_shape_bins   = list(range(5, 10)) # rounded
    size_bins = list(range(10))               # columns

    # small/large by size (columns)
    small_size_bins = list(range(0, 5))   # bins 0..4
    large_size_bins = list(range(5, 10))  # bins 5..9

    # grouped 5x5 blocks for shape × size
    grouped_blocks = [
    (list(range(0,5)), list(range(0,5))),  # top-left block: elongated shapes × small sizes (0-4)x(0-4)
    (list(range(0,5)), list(range(5,10))), # top-right: elongated shapes × large sizes (0-4)x(5-9)
    (list(range(5,10)), list(range(0,5))), # bottom-left: compact shapes × small sizes (5-9)x(0-4)
    (list(range(5,10)), list(range(5,10))) # bottom-right: compact shapes × large sizes (5-9)x(5-9)
    ]

    subset_specs = []

    # 1) small sizes
    subset_specs.append({
        "kind": "small_sizes",
        "shape_bins": less_compact_shape_bins + more_compact_shape_bins,  
        "size_bins": small_size_bins
    })
    # 2) large sizes
    subset_specs.append({
        "kind": "large_sizes",
        "shape_bins": less_compact_shape_bins + more_compact_shape_bins,  
        "size_bins": large_size_bins
    })
    # 3) shape groups
    subset_specs.append({
        "kind": "elongated_shapes",
        "shape_bins": less_compact_shape_bins,
        "size_bins": size_bins.copy()
    })
    subset_specs.append({
        "kind": "compact_shapes",
        "shape_bins": more_compact_shape_bins,
        "size_bins": size_bins.copy()
    })
    # 4) grouped blocks
    for i, (s_bins, z_bins) in enumerate(grouped_blocks):
        subset_specs.append({
            "kind": f"block_{i}",
            "shape_bins": s_bins,
            "size_bins": z_bins
        })


    print(f"Will evaluate {len(subset_specs)} subsets per channel. Subset kinds example: {[s['kind'] for s in subset_specs[:6]]} ...")

    # For each channel in best_combo, run the bin-analysis
    for ch in best_combo:
        print("\n" + "="*60)
        print(f"Running bin-analysis for variant={VARIANT}, channel={ch}")
        X_train_per_label = []
        y_train_per_label = []
        X_test_per_label = []
        y_test_per_label = []

        for label_idx, (label_name, prefix) in enumerate(label_map.items()):
            Xtr_arrs = preloaded[label_name][ch + "_train"]   
            Xte_arrs = preloaded[label_name][ch + "_test"]
            ntr = Xtr_arrs.shape[0]
            nte = Xte_arrs.shape[0]
            X_train_per_label.append(Xtr_arrs)
            y_train_per_label.append(np.array([label_idx]*ntr))
            X_test_per_label.append(Xte_arrs)
            y_test_per_label.append(np.array([label_idx]*nte))

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

            # Build feature vectors for train and test; if a sample's selected block is all-zero, skip it
            Xtr_list = []
            ytr_list = []
            for xi, yi in zip(X_train_full_10x10, y_train_full):
                fv = create_feature_array_from_10x10(xi, s_bins, z_bins)
                if fv is None:
                    continue
                Xtr_list.append(fv)
                ytr_list.append(yi)
            Xtr_list = np.array(Xtr_list)
            ytr_list = np.array(ytr_list, dtype=np.int64)

            Xte_list = []
            yte_list = []
            for xi, yi in zip(X_test_full_10x10, y_test_full):
                fv = create_feature_array_from_10x10(xi, s_bins, z_bins)
                if fv is None:
                    continue
                Xte_list.append(fv)
                yte_list.append(yi)
            Xte_list = np.array(Xte_list)
            yte_list = np.array(yte_list, dtype=np.int64)

            # If too few samples to do CV, skip subset (but save that it's invalid)
            if Xtr_list.shape[0] < 2:
                print(f"  Skipping {kind}: not enough training samples after removing zero blocks (n={Xtr_list.shape[0]})")
                channel_results.append({
                    "kind": kind,
                    "note": "insufficient_train_samples",
                    "n_train_after_filter": int(Xtr_list.shape[0]),
                    "n_test_after_filter": int(Xte_list.shape[0])
                })
                continue

            iter_metrics = []
            for seed in range(iterations):
                np.random.seed(seed)
                splitter = random_splitter_gen(Xtr_list, ytr_list, xval_fraction)
                f1s, accs, precs, recs = [], [], [], []
                confs = []
                for fold in range(xval_count):
                    x_train, y_train_fold, x_val, y_val_fold = next(splitter)
                    # evaluate fold
                    f1, acc, prec, rec, conf = evaluate_fold(x_train, y_train_fold, x_val, y_val_fold)
                    f1s.append(f1); accs.append(acc); precs.append(prec); recs.append(rec); confs.append(conf)
                iter_metrics.append({
                    "seed": seed,
                    "f1_mean": float(np.mean(f1s)),
                    "acc_mean": float(np.mean(accs)),
                    "prec_mean": float(np.mean(precs)),
                    "rec_mean": float(np.mean(recs)),
                    "conf_mean": np.mean(confs, axis=0).tolist()
                })
                print(f"   iteration {seed}: F1={np.mean(f1s):.4f}, Acc={np.mean(accs):.4f}, Prec={np.mean(precs):.4f}, Rec={np.mean(recs):.4f}")

            # aggregate across iterations
            f1_by_iter = np.array([it["f1_mean"] for it in iter_metrics])
            acc_by_iter = np.array([it["acc_mean"] for it in iter_metrics])
            prec_by_iter = np.array([it["prec_mean"] for it in iter_metrics])
            rec_by_iter = np.array([it["rec_mean"] for it in iter_metrics])

            result_entry = {
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
            }

            channel_results.append(result_entry)

        # save intermediate results for this channel
        out_pkl = os.path.join(OUT_DIR, f"{VARIANT}_bin_analysis_{ch}.pkl")
        with open(out_pkl, "wb") as f:
            pickle.dump(channel_results, f)
        print(f"\nSaved bin-analysis results for channel {ch} to {out_pkl}")

        # Find best subset by f1_mean_across_iters
        valid_results = [r for r in channel_results if "f1_mean_across_iters" in r]
        if not valid_results:
            print(f"No valid subsets for channel {ch} (all skipped). Continuing to next channel.")
            continue
        best_subset = max(valid_results, key=lambda rr: rr["f1_mean_across_iters"])
        print(f"\nBest subset for channel {ch}: {best_subset['kind']} (avg F1={best_subset['f1_mean_across_iters']:.4f}, std={best_subset['f1_std_across_iters']:.4f})")

        # Retrain final model on full train set (for that subset) and evaluate on the held-out test set
        s_bins = best_subset["shape_bins"]
        z_bins = best_subset["size_bins"]

        # Construct final train/test feature matrices (exclude samples whose selected block is zero)
        Xtr_final_list = []
        ytr_final_list = []
        for xi, yi in zip(X_train_full_10x10, y_train_full):
            fv = create_feature_array_from_10x10(xi, s_bins, z_bins)
            if fv is None:
                continue
            Xtr_final_list.append(fv); ytr_final_list.append(yi)
        Xtr_final = np.array(Xtr_final_list); ytr_final = np.array(ytr_final_list, dtype=np.int64)

        Xte_final_list = []
        yte_final_list = []
        for xi, yi in zip(X_test_full_10x10, y_test_full):
            fv = create_feature_array_from_10x10(xi, s_bins, z_bins)
            if fv is None:
                continue
            Xte_final_list.append(fv); yte_final_list.append(yi)
        Xte_final = np.array(Xte_final_list); yte_final = np.array(yte_final_list, dtype=np.int64)

        if Xtr_final.shape[0] < 2 or Xte_final.shape[0] < 1:
            print(f"Not enough data to retrain/test for best subset {best_subset['kind']} (train n={Xtr_final.shape[0]}, test n={Xte_final.shape[0]}). Skipping final eval.")
            continue

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
        final_model.fit(Xtr_final, ytr_final)
        y_pred_test = final_model.predict(Xte_final)

        final_f1 = f1_score(yte_final, y_pred_test, average='weighted')
        final_acc = accuracy_score(yte_final, y_pred_test)
        final_prec = precision_score(yte_final, y_pred_test, average='weighted', zero_division=0)
        final_rec = recall_score(yte_final, y_pred_test, average='weighted', zero_division=0)
        final_conf = confusion_matrix(yte_final, y_pred_test, normalize='true')

        print("\n===== FINAL TEST SET RESULTS (best subset) =====")
        print(f"Channel {ch} | Best subset {best_subset['kind']}")
        print(f"Test Weighted F1: {final_f1:.4f}")
        print(f"Test Accuracy   : {final_acc:.4f}")
        print(f"Test Precision  : {final_prec:.4f}")
        print(f"Test Recall     : {final_rec:.4f}")
        print("Test Confusion Matrix (normalized):")
        print(final_conf)

        # save model + confusion matrix + best subset description
        model_path = os.path.join(OUT_DIR, f"{VARIANT}_{ch}_best_subset_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(final_model, f)
        save_confusion_matrix(final_conf, out_dir=OUT_DIR, name=f"{VARIANT}_{ch}_best_subset_conf_matrix")
        meta = {
            "variant": VARIANT,
            "channel": ch,
            "best_subset": best_subset,
            "final_metrics": {
                "f1": float(final_f1),
                "acc": float(final_acc),
                "prec": float(final_prec),
                "rec": float(final_rec)
            },
            "model_path": model_path
        }
        with open(os.path.join(OUT_DIR, f"{VARIANT}_{ch}_best_subset_meta.pkl"), "wb") as f:
            pickle.dump(meta, f)

        print(f"Saved final model to {model_path} and metadata.")

    print("\nAll channels processed. Bin-analysis complete.")

if __name__ == "__main__":
    main()
