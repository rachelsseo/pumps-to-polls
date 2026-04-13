import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os
from pymongo import MongoClient

# 1. pulling data
client = MongoClient(os.getenv("MONGO_URI"))
collection = client["data_by_design"]["election_economics"]

df = pd.DataFrame(list(collection.find({}, {"_id": 0})))
print(f"Loaded {len(df)} documents")
print(df.head())

# 2. aggregating to national level
national = df.groupby(["year", "incumbent_party", "gas_price_change_pct", "inflation_rate"]) \
             .agg(avg_vote_share=("incumbent_vote_share", "mean")) \
             .reset_index()

national = national.sort_values("year")
print(national)

# 3. chart
fig, ax1 = plt.subplots(figsize=(13, 6))

# color bars by party
colors = ["#E63946" if p == "REPUBLICAN" else "#457B9D"
          for p in national["incumbent_party"]]

bars = ax1.bar(national["year"], national["avg_vote_share"],
               color=colors, width=2.5, alpha=0.85, label="Incumbent vote share")

ax1.set_xlabel("Election Year", fontsize=12)
ax1.set_ylabel("Avg Incumbent Vote Share (%)", fontsize=12)
ax1.set_ylim(30, 70)
ax1.axhline(50, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
ax1.set_xticks(national["year"])
ax1.set_xticklabels(national["year"], rotation=45)

# overlay gas price change line
ax2 = ax1.twinx()
ax2.plot(national["year"], national["gas_price_change_pct"],
         color="darkorange", marker="o", linewidth=2,
         markersize=6, label="Gas price change (%)")
ax2.plot(national["year"], national["inflation_rate"],
         color="green", marker="s", linewidth=2, linestyle="--",
         markersize=6, label="Inflation rate (%)")
ax2.set_ylabel("% Change", fontsize=12)
ax2.axhline(0, color="grey", linewidth=0.5, linestyle=":")

# labels on bars
for bar, val in zip(bars, national["avg_vote_share"]):
    ax1.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.5,
             f"{val:.1f}%", ha="center", va="bottom", fontsize=8)

# legend
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

legend_elements = [
    Patch(facecolor="#E63946", alpha=0.85, label="Republican incumbent"),
    Patch(facecolor="#457B9D", alpha=0.85, label="Democrat incumbent"),
    Line2D([0], [0], color="darkorange", marker="o", label="Gas price change (%)"),
    Line2D([0], [0], color="green", marker="s", linestyle="--", label="Inflation rate (%)")
]
ax1.legend(handles=legend_elements, loc="upper left", fontsize=9)

plt.title("At the Pump and at the Polls\nIncumbent Vote Share vs. Gas Prices & Inflation (1976–2020)",
          fontsize=13, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig("press_release_chart.png", dpi=150, bbox_inches="tight")
plt.show()
print("Chart saved to press_release_chart.png")