import os
import pickle
import numpy as np
from scipy.stats import f_oneway

RESULT_PATH = "Channel_Combo_Results/Fontane_channel_results.pkl"
with open(RESULT_PATH, "rb") as f:
    results = pickle.load(f)

# sort by average F1 and take top 5 
results_sorted = sorted(results, key=lambda x: x["avg_f1"], reverse=True)
top5 = results_sorted[:5]

f1_groups = []
for i, r in enumerate(top5):
    print(f"Condition {i+1}: {r['combo']}, avg F1 = {r['avg_f1']:.4f}")
    f1_groups.append(np.array(r["f1_scores"]))

#  one-way ANOVA 
F_stat, p_value = f_oneway(*f1_groups)

print("\nOne-way ANOVA results:")
print(f"F = {F_stat:.4f}")
print(f"p = {p_value:.6f}")

if p_value < 0.05:
    print("Significant difference between conditions")
else:
    print("No significant difference detected")
