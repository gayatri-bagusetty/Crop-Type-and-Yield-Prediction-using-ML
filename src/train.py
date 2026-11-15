import pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("data/crop_yield.csv")
for c in ["Crop", "Season", "State"]:
    df[c] = df[c].str.strip()          # fix trailing spaces
df = df.drop(columns=["Production"])   # remove the leak

X, y = df.drop(columns=["Yield"]), df["Yield"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

pre = ColumnTransformer(
    [("cat", OneHotEncoder(handle_unknown="ignore"), ["Crop", "Season", "State"])],
    remainder="passthrough",
)
model = Pipeline([("pre", pre), ("rf", RandomForestRegressor(200, random_state=42, n_jobs=-1))])
model.fit(X_tr, y_tr)
joblib.dump(model, "model.joblib")


from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np

pred = model.predict(X_te)
print("R2:", r2_score(y_te, pred))
print("MAE:", mean_absolute_error(y_te, pred))
print("Median abs error:", np.median(np.abs(y_te - pred)))