import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

sns.set_theme(style="whitegrid")
df = pd.read_csv("data/crop_yield.csv")
for c in ["Crop", "Season", "State"]:
    df[c] = df[c].str.strip()
df["Fert_per_ha"] = df["Fertilizer"] / df["Area"]
df["Pest_per_ha"] = df["Pesticide"] / df["Area"]

# NOTE: yield units differ by crop (coconut is per nuts), so compare WITHIN one crop.
rice = df[df.Crop == "Rice"]

# 1) Yield trend over years for 4 major crops
major = ["Rice", "Wheat", "Maize", "Sugarcane"]
t = df[df.Crop.isin(major)].groupby(["Crop_Year", "Crop"]).Yield.median().reset_index()
g = sns.relplot(data=t, x="Crop_Year", y="Yield", col="Crop", col_wrap=2, kind="line",
                facet_kws={"sharey": False}, height=3, aspect=1.5)
g.fig.suptitle("Median yield over time (major crops)", y=1.02)
g.savefig("results/01_yield_trend.png", dpi=150, bbox_inches="tight")
print("Median yield first vs last 3 years:")
for c in major:
    s = df[df.Crop == c]
    a = s[s.Crop_Year <= s.Crop_Year.min() + 2].Yield.median()
    b = s[s.Crop_Year >= s.Crop_Year.max() - 2].Yield.median()
    print(f"  {c}: {a:.2f} -> {b:.2f}  ({100*(b-a)/a:+.0f}%)")

# 2) Rice yield by state (top 10 states by median)
top = rice.groupby("State").Yield.median().sort_values(ascending=False).head(10)
plt.figure(figsize=(8, 4)); sns.barplot(x=top.values, y=top.index, color="#4394E5")
plt.title("Rice: median yield by state (top 10)"); plt.xlabel("Yield"); plt.tight_layout()
plt.savefig("results/02_rice_by_state.png", dpi=150); plt.close()
print("\nRice top 3 states:", top.head(3).round(2).to_dict())
print("Rice lowest state (>=20 rows):",
      rice.groupby("State").Yield.agg(["median", "size"]).query("size>=20")["median"].sort_values().head(1).round(2).to_dict())

# 3) Rice yield by season
plt.figure(figsize=(7, 4)); sns.boxplot(data=rice, x="Season", y="Yield", color="#87BB62")
plt.title("Rice: yield by season"); plt.tight_layout()
plt.savefig("results/03_rice_by_season.png", dpi=150); plt.close()
print("\nRice median yield by season:\n", rice.groupby("Season").Yield.agg(["median", "size"]).round(2))

# 4) Fertilizer per hectare vs yield (Rice)
plt.figure(figsize=(6, 4.5))
sns.scatterplot(data=rice, x="Fert_per_ha", y="Yield", alpha=0.4)
plt.xscale("log"); plt.title("Rice: fertilizer per hectare vs yield"); plt.tight_layout()
plt.savefig("results/04_rice_fertilizer.png", dpi=150); plt.close()
print("\nRice Spearman corr  Fert/ha vs Yield: %.2f | Rainfall vs Yield: %.2f" % (
    rice.Fert_per_ha.corr(rice.Yield, method="spearman"),
    rice.Annual_Rainfall.corr(rice.Yield, method="spearman")))

# 5) Rainfall vs yield (Rice)
plt.figure(figsize=(6, 4.5))
sns.scatterplot(data=rice, x="Annual_Rainfall", y="Yield", alpha=0.4)
plt.title("Rice: annual rainfall vs yield"); plt.tight_layout()
plt.savefig("results/05_rice_rainfall.png", dpi=150); plt.close()

# 6) Permutation importance of the leak-free model
d = df.drop(columns=["Production", "Fert_per_ha", "Pest_per_ha"])
X, y = d.drop(columns=["Yield"]), d["Yield"]
cat = ["Crop", "Season", "State"]; num = [c for c in X.columns if c not in cat]
pipe = Pipeline([("pre", ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore"), cat),
                                            ("n", StandardScaler(), num)])),
                 ("m", RandomForestRegressor(200, random_state=42, n_jobs=-1))])
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
pipe.fit(X_tr, y_tr)
imp = permutation_importance(pipe, X_te, y_te, n_repeats=5, random_state=42, n_jobs=-1)
imp_s = pd.Series(imp.importances_mean, index=X.columns).sort_values()
plt.figure(figsize=(6, 4)); imp_s.plot.barh(color="#F5921B")
plt.title("Permutation importance (drop in R2)"); plt.tight_layout()
plt.savefig("results/06_importance.png", dpi=150); plt.close()
print("\nPermutation importance:\n", imp_s.sort_values(ascending=False).round(3))