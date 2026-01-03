import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(".."))

with open("Saved_Models/Exp1/Mondial_best_model-ALL.pkl", "rb") as f:
    model = pickle.load(f)

omegas = model.omegas_  # list of omegas per prototype

lambda_diagonals = []
for omega in omegas:
    Lambda = omega.T @ omega
    lambda_diagonals.append(np.diag(Lambda))
lambda_diagonals = np.array(lambda_diagonals)

features_per_channel = lambda_diagonals.shape[1] // 6
proto_classes = model.c_w_ 


class0_relevance = lambda_diagonals[proto_classes == 0].mean(axis=0)
class1_relevance = lambda_diagonals[proto_classes == 1].mean(axis=0)

channels = ["R", "G", "B", "H", "S", "V"]
features_per_channel = lambda_diagonals.shape[1] // len(channels)

COLOR_MAP = {
    "R": "#E96060",     
    "B": "#4EA7FF",      
    "G": "#98D798",      
    "H": "#EC7FCA",      
    "V": "#F3BA6B",      
    "S": "#792468",      
}

def darken_hex(hex_color, factor=0.7):
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    r = int(r * factor)
    g = int(g * factor)
    b = int(b * factor)
    return f"#{r:02x}{g:02x}{b:02x}"

plt.figure(figsize=(12, 5))

start = 0
for ch in channels:
    end = start + features_per_channel
    base_color = COLOR_MAP.get(ch, "black")
    darker_color = darken_hex(base_color, factor=0.6)

    # class 0 solid line
    plt.plot(
        range(start, end),
        class0_relevance[start:end],
        label=f"{ch} – healthy",
        color=base_color,
        linestyle="-",
        linewidth=2
    )
    # class 1 dashed line (
    plt.plot(
        range(start, end),
        class1_relevance[start:end],
        label=f"{ch} – diseased",
        color=darker_color,
        linestyle="--",
        linewidth=2
    )
    start = end

plt.xlabel("Feature index")
plt.ylabel("Relevance (diag($\\Lambda$))")
plt.title("Class-wise feature relevance (IAALVQ)")
plt.legend(ncol=2, fontsize=8)
plt.tight_layout()
plt.show()
