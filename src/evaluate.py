import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# ---- load and clean ----
df = pd.read_csv("data/crop_yield.csv")
for c in ["Crop", "Season", "State"]:
    df[c] = df[c].str.strip()
df = df.drop(columns=["Production"])          # remove the leak

X, y = df.drop(columns=["Yield"]), df["Yield"]
cat = ["Crop", "Season", "State"]
num = [c for c in X.columns if c not in cat]

def make(model):
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat),
                             ("num", StandardScaler(), num)])
    return Pipeline([("pre", pre), ("m", model)])

models = {
    "Baseline (mean)": DummyRegressor(),
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(200, random_state=42, n_jobs=-1),
    "XGBoost": XGBRegressor(n_estimators=300, learning_rate=0.1, random_state=42, n_jobs=-1),
}

# ---- 5-fold cross-validation ----
kf = KFold(5, shuffle=True, random_state=42)
rows = []
for name, m in models.items():
    s = cross_validate(make(m), X, y, cv=kf, scoring=["r2", "neg_mean_absolute_error"])
    rows.append({"Model": name,
                 "R2 (mean)": s["test_r2"].mean(),
                 "R2 (std)": s["test_r2"].std(),
                 "MAE (mean)": -s["test_neg_mean_absolute_error"].mean()})
results = pd.DataFrame(rows).round(3)
print(results.to_string(index=False))
results.to_csv("results/yield_cv_results.csv", index=False)

# ---- per-crop error for the Random Forest (hold-out set) ----
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
best = make(RandomForestRegressor(200, random_state=42, n_jobs=-1)).fit(X_tr, y_tr)
pred = best.predict(X_te)
print("\nHold-out  R2: %.3f  MAE: %.2f  Median abs err: %.2f" %
      (r2_score(y_te, pred), mean_absolute_error(y_te, pred), np.median(np.abs(y_te - pred))))

d = X_te.copy(); d["abs_err"] = np.abs(y_te - pred); d["actual"] = y_te
per_crop = (d.groupby("Crop").agg(n=("abs_err", "size"), MAE=("abs_err", "mean"),
                                  median_yield=("actual", "median"))
              .assign(MAE_pct_of_median=lambda t: 100 * t.MAE / t.median_yield)
              .round(2).sort_values("MAE_pct_of_median"))
per_crop.to_csv("results/per_crop_error.csv")
print("\nWorst 5 crops by relative error:\n", per_crop.tail(5))
print("\nBest 5 crops by relative error:\n", per_crop.head(5))