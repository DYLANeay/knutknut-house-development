import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# this script is in report/, the model code is one folder up (kkd/)
HERE = os.path.dirname(os.path.abspath(__file__))
KKD_DIR = os.path.dirname(HERE)
sys.path.append(KKD_DIR)

from kkd_data import FEATURE_NAMES
from model import load_model

# readable names for the presentation
LABELS = {
    "size": "Living size",
    "lot_w": "Lot width",
    "external_storage_m2": "External storage (m²)",
    "storage_rating": "Storage rating",
    "condition_rating": "Condition",
    "bathrooms": "Bathrooms",
    "was_remodeled": "Was remodeled",
    "kitchens": "Kitchens",
    "school_rating": "School rating",
    "years_since_remodel": "Years since remodel",
    "ad_premium": "Premium ad",
    "ad_regular": "Regular ad",
    "age": "Age",
    "sun_factor": "Sun factor",
    "school_capacity": "School capacity",
    "fireplace": "Fireplace",
    "parking": "Parking",
}

weights, bias = load_model(os.path.join(KKD_DIR, "model.npz"))
names = [LABELS.get(n, n.replace("_", " ").capitalize()) for n in FEATURE_NAMES]

# biggest weights at the top, keep the 12 most important
order = np.argsort(np.abs(weights))[-12:]
values = weights[order]
labels = [names[i] for i in order]
colors = ["#1e3a8a" if v > 0 else "#dc2626" for v in values]

fig, ax = plt.subplots(figsize=(12, 7))
bars = ax.barh(labels, values, color=colors)
ax.bar_label(bars, labels=[f"{v:+.2f}" for v in values], padding=5, fontsize=13)

ax.set_title("What drives the price of a house", fontsize=20, fontweight="bold", loc="left")
ax.set_xlabel("Effect on price (million NOK per standard deviation)", fontsize=14)
ax.tick_params(axis="y", labelsize=14)
ax.tick_params(axis="x", labelsize=12)
ax.axvline(0, color="black", linewidth=0.8)
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlim(-1.0, values.max() * 1.15)   # room on the left for the negative labels

plt.tight_layout()
plt.savefig(os.path.join(HERE, "key_drivers.png"), dpi=200)
print("saved report/key_drivers.png")
