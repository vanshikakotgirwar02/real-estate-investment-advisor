import numpy as np
import pandas as pd

# ---------- 1. Load ----------
df = pd.read_csv("data/india_housing_prices.csv")
print("Loaded:", df.shape)

# ---------- 2. Duplicates ----------
before = len(df)
df = df.drop_duplicates(subset=[c for c in df.columns if c != "ID"])
print("Duplicates removed:", before - len(df))

# ---------- 3. Missing values ----------
print("Missing values:", df.isna().sum().sum())
for c in df.select_dtypes("number").columns:
    df[c] = df[c].fillna(df[c].median())          # numbers -> median
for c in df.select_dtypes(exclude="number").columns:
    df[c] = df[c].fillna(df[c].mode().iloc[0])    # text -> most common value

# ---------- 4. Convert Yes/No columns to 1/0 ----------
df["Parking_Space"] = df["Parking_Space"].map({"Yes": 1, "No": 0})

# ---------- 5. Outliers (IQR rule) ----------
pps = np.log(df["Price_in_Lakhs"] * 1e5 / df["Size_in_SqFt"])   # price per sq ft in Rs, log scale
q1, q3 = pps.quantile([0.25, 0.75])
iqr = q3 - q1
is_outlier = ~pps.between(q1 - 3 * iqr, q3 + 3 * iqr)
print("Outliers found:", is_outlier.sum(), f"({is_outlier.mean():.2%})")
df = df[~is_outlier]

# ---------- 6. Features + targets ----------
import json
from features import compute_stats, engineer, make_targets

stats = compute_stats(df)
df = make_targets(engineer(df, stats), stats)

df.to_csv("data/housing_processed.csv", index=False)
json.dump(stats, open("models/stats.json", "w"), indent=2)
print("Saved:", df.shape)
print("Good investment share:", round(df["Good_Investment"].mean(), 3))
print("Median growth rate:", round(df["Growth_Rate"].median(), 4))
print(df[["Price_in_Lakhs", "Growth_Rate", "Future_Price_Lakhs", "Invest_Score", "Good_Investment"]].head())