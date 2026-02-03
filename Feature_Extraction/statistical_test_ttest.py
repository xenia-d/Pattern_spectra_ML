import pickle
import numpy as np
from scipy.stats import ttest_ind


VARIANT = "Fontane"
RESULT_PATH = f"Channel_Combo_Results/{VARIANT}_channel_results.pkl"

print("Variant:", {VARIANT})

with open(RESULT_PATH, "rb") as f:
    results = pickle.load(f)

top_entry = max(results, key=lambda x: x["avg_f1"])
rgb_entry = None
hsv_entry = None

for r in results:
    if r["combo"] == "R_G_B":
        rgb_entry = r
    elif r["combo"] == "H_S_V":
        hsv_entry = r

top_f1 = np.array(top_entry["f1_scores"])
rgb_f1 = np.array(rgb_entry["f1_scores"])
hsv_f1 = np.array(hsv_entry["f1_scores"])

print(f"Top-performing: {top_entry['combo']} (avg F1 = {top_entry['avg_f1']:.4f})")
print(f"RGB:            {rgb_entry['combo']} (avg F1 = {rgb_entry['avg_f1']:.4f})")
print(f"HSV:            {hsv_entry['combo']} (avg F1 = {hsv_entry['avg_f1']:.4f})")

# top vs RGB 
t_rgb, p_rgb = ttest_ind(top_f1, rgb_f1, equal_var=False)
print("\nTop vs RGB:")
print(f"t = {t_rgb:.4f}, p = {p_rgb:.6f}")
if p_rgb < 0.05:
    print("Significant difference (p < 0.05)")
else:
    print("No significant difference (p ≥ 0.05)")

# top vs HSV 
t_hsv, p_hsv = ttest_ind(top_f1, hsv_f1, equal_var=False)
print("\nTop vs HSV:")
print(f"t = {t_hsv:.4f}, p = {p_hsv:.6f}")
if p_hsv < 0.05:
    print("Significant difference")
else:
    print("No significant difference")
