
from lvq.GIALVQ import GIALVQ
from lvq.IAALVQ import IAALVQ
import torch
import glob
from utils.io_management import *
from utils.preprocessing import *
from utils.segmentation import build_informed_tree, cut_tree, segment_gialvq
from utils.visualization import *
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
import cv2
import numpy as np
from collections import defaultdict
from skimage import exposure

def load_granulometry_m_file(path):
    data = []
    with open(path, 'r') as f:
        lines = f.readlines()
    lines = lines[2:]  # skip header + column indices
    for line in lines:
        line = line.strip()
        if not line:
            continue
        tokens = line.split()
        numbers = [float(x) for x in tokens[1:]]  # skip row index
        data.append(numbers)
    return np.array(data)
def create_rgb_feature_vector(R_file, G_file, B_file, H_file, S_file, V_file):
    R = np.array(load_granulometry_m_file(R_file))[:10,:10].flatten()
    if np.max(R) == 0:
        return None
    # R= R/ np.max(R)
    # R= np.log1p/(R)
    G = np.array(load_granulometry_m_file(G_file))[:10,:10].flatten()
    if np.max(G) == 0:
        return None
    # G= G/ np.max(G)
    # G= np.log1p(G)  
    B = np.array(load_granulometry_m_file(B_file))[:10,:10].flatten()
    if np.max(B) == 0:
        return None
    # # 
    # B= B/ np.max(B)
    # B= np.log1p(B)
    H = np.array(load_granulometry_m_file(H_file))[:10,:10].flatten()
    if np.max(H) == 0:
        return None
    # H= H/ np.max(H)
    # H= np.log1p(H)
    # Make sure they are the same length
    if  len(R) != len(G) != len(B):
        raise ValueError("R, G, B arrays must be the same length")
    # Stack into N x 3 array
    rgb_features = np.concatenate([R, G, B, H ], axis=0)
    return rgb_features
# To generate training and validation sets

def random_splitter_gen(imgs, labels,  validation_fraction):     
    # internal state
    state = np.random.get_state()
    
    while (True):
        # save current state
        global_state = np.random.get_state()
        # load internal state
        np.random.set_state(state)
        
        I = np.random.permutation(len(imgs))
        n_val = np.int32(np.round(len(imgs)) * validation_fraction)
    
        imgs_val = [imgs[i] for i in I[0:n_val]]
        labels_val = labels[I[0:n_val]]
        # varieties_val = [varieties[i] for i in I[0:n_val]]
    
        imgs_train = [imgs[i] for i in I[n_val:]]
        labels_train = labels[I[n_val:]]
        # varieties_train = [varieties[i] for i in I[n_val:]]

        # save internal state
        state = np.random.get_state()
        # restore state
        np.random.set_state(global_state)

        yield imgs_train, labels_train, imgs_val, labels_val

def main():
    R_files = sorted(glob.glob(r"C:\Users\anush\Documents\PostDoc\Croptimal datasets\NAKFielddataset\Spunta_variety\leaf_images\RGBFilter\channels_pgm\features\hR*.m"))
    G_files = sorted(glob.glob(r"C:\Users\anush\Documents\PostDoc\Croptimal datasets\NAKFielddataset\Spunta_variety\leaf_images\RGBFilter\channels_pgm\features\hG*.m"))
    B_files = sorted(glob.glob(r"C:\Users\anush\Documents\PostDoc\Croptimal datasets\NAKFielddataset\Spunta_variety\leaf_images\RGBFilter\channels_pgm\features\hB*.m"))
    H_files = sorted(glob.glob(r"C:\Users\anush\Documents\PostDoc\Croptimal datasets\NAKFielddataset\Spunta_variety\leaf_images\RGBFilter\channels_pgm\features\hS*.m"))
    import random
    random.seed(2)  # ← set a fixed seed (any number works)
    # num_samples = 2500
    R_indices = range(len(R_files))
    R_files = [R_files[i] for i in R_indices]
    G_files = [G_files[i] for i in R_indices]
    B_files = [B_files[i] for i in R_indices]
    H_files = [H_files[i] for i in R_indices]
    all_histogramsH = []#[] for _ in range(num_bins)]  # Pre-create 121 bins
    for i in range(len(R_files)):
        R_file = R_files[i]
        G_file = G_files[i]
        B_file = B_files[i]
        H_file = H_files[i]
        rgb_feature = create_rgb_feature_vector(R_file, G_file, B_file, H_file,  None, None)
        if rgb_feature is not None:
            all_histogramsH.append(rgb_feature)
    import numpy as np
    all_histogramsH = np.array(all_histogramsH)
    R_files = sorted(glob.glob(r"C:\Users\anush\Documents\PostDoc\Croptimal datasets\NAKFielddataset\Spunta_variety\leaf_images\RGBFilter\channels_pgm\features\uhR*.m"))
    G_files = sorted(glob.glob(r"C:\Users\anush\Documents\PostDoc\Croptimal datasets\NAKFielddataset\Spunta_variety\leaf_images\RGBFilter\channels_pgm\features\uhG*.m"))
    B_files = sorted(glob.glob(r"C:\Users\anush\Documents\PostDoc\Croptimal datasets\NAKFielddataset\Spunta_variety\leaf_images\RGBFilter\channels_pgm\features\uhB*.m"))
    H_files = sorted(glob.glob(r"C:\Users\anush\Documents\PostDoc\Croptimal datasets\NAKFielddataset\Spunta_variety\leaf_images\RGBFilter\channels_pgm\features\uhS*.m"))
    random.seed(4)
    num_samples = len(B_files)
    R_indices = range(len(B_files))
    print(R_indices)
    R_files = [R_files[i] for i in R_indices]
    G_files = [G_files[i] for i in R_indices]
    B_files = [B_files[i] for i in R_indices]
    H_files = [H_files[i] for i in R_indices]
    all_histogramsnhy = []
    for i in range(len(R_files)):
        R_file = R_files[i]
        G_file = G_files[i]
        B_file = B_files[i]
        H_file = H_files[i]
        # S_file = S_files[i]
        # V_file = V_files[i]
        rgb_feature = create_rgb_feature_vector(R_file, G_file, B_file, H_file, None, None)
        if rgb_feature is not None:
            all_histogramsnhy.append(rgb_feature)
    a2_flat = np.array(all_histogramsnhy).reshape(-1, 100*4)
    b_flat = np.array(all_histogramsH).reshape(-1, 100*4)
    X1 = np.vstack([b_flat, a2_flat])
    y1 = np.array([0]*len(b_flat) + [1]*len(a2_flat))
    xval_count = 10    
    xval_fraction = 0.2
    random_splitter = random_splitter_gen(X1, y1, xval_fraction)

    for xval_nr in range(xval_count):
        x_train, y_train,  x_test, y_test = next(random_splitter)
        model = IAALVQ(max_iter=50, prototypes_per_class=2, omega_rank=400, seed=59,
                regularization=0.00001, omega_locality='PW', filter_bank=None,
                block_eye=False, norm=False, correct_imbalance=True)
    # X1[np.all(X1 == 0, axis=1), :] += 1e-12  
    # Split into train and test
    # from sklearn.model_selection import train_test_split
    # x_train, x_test, y_train, y_test = train_test_split(
    #     X1, y1, test_size=.2, shuffle=True, random_state=42, stratify=y1
    # )
    # with open("ialvq_modelSpunta.pkl", "wb") as f:
    # pickle.dump(model, f)
        x_train = np.array(x_train)
        x_test = np.array(x_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        model.fit(x_train, y_train)
  
        y_pred_test = model.predict(x_test)
        y_pred_train = model.predict(x_train)
        f1_test= f1_score(y_test, y_pred_test, average='weighted')
        f1_train= f1_score(y_train, y_pred_train, average='weighted')
        print("Train F1-score: ", f1_train)
        print("Test F1-score: ", f1_test)
        print("Train accuracy: ", model.score(x_train, y_train))
        print("Test accuracy: ", model.score(x_test, y_test))
        print(confusion_matrix(y_test, y_pred_test, normalize= "true"))

if __name__ == "__main__":
    main()
