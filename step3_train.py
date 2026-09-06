import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing   import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model    import LinearRegression
from sklearn.ensemble        import RandomForestRegressor
from sklearn.metrics         import mean_squared_error, mean_absolute_error, r2_score

os.makedirs("models", exist_ok=True)
os.makedirs("plots",  exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 11, "axes.titlesize": 13})

print("Loading clean dataset...")
df = pd.read_csv("data/autos_clean.csv")
print(f"  {len(df):,} rows, {df.shape[1]} columns")

print("\n[1] Encoding categorical columns...")

CATEGORICAL_COLS = ["brand", "model", "color", "transmission_type", "fuel_type"]

encoders = {}

for col in CATEGORICAL_COLS:
    le             = LabelEncoder()
    df[col]        = le.fit_transform(df[col].astype(str))
    encoders[col]  = le
    print(f"    {col}: {len(le.classes_)} unique values encoded")

FEATURES = ["brand", "model", "color", "year", "power_kw",
            "transmission_type", "fuel_type", "mileage_in_km"]
TARGET   = "price_in_euro"

X = df[FEATURES]
y = df[TARGET]

print(f"\n[2] Features: {FEATURES}")
print(f"    Target  : {TARGET}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n[3] Train/test split:")
print(f"    Training rows : {len(X_train):,}")
print(f"    Test rows     : {len(X_test):,}")

print("\n[4] Training Linear Regression (baseline)...")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

lr_preds = lr_model.predict(X_test)
lr_rmse  = np.sqrt(mean_squared_error(y_test, lr_preds))
lr_mae   = mean_absolute_error(y_test, lr_preds)
lr_r2    = r2_score(y_test, lr_preds)

print(f"    RMSE : €{lr_rmse:,.0f}")
print(f"    MAE  : €{lr_mae:,.0f}")
print(f"    R²   : {lr_r2:.4f}")

print("\n[5] Training Random Forest (this takes 2–4 minutes)...")
rf_model = RandomForestRegressor(
    n_estimators=100,
    n_jobs=-1,
    random_state=42
)
rf_model.fit(X_train, y_train)

rf_preds = rf_model.predict(X_test)
rf_rmse  = np.sqrt(mean_squared_error(y_test, rf_preds))
rf_mae   = mean_absolute_error(y_test, rf_preds)
rf_r2    = r2_score(y_test, rf_preds)

print(f"    RMSE : €{rf_rmse:,.0f}")
print(f"    MAE  : €{rf_mae:,.0f}")
print(f"    R²   : {rf_r2:.4f}")

print("\n[6] Model comparison:")
print(f"\n  {'Metric':<8} {'Linear Regression':>20} {'Random Forest':>20}")
print(f"  {'-'*50}")
print(f"  {'RMSE':<8} {'€' + f'{lr_rmse:,.0f}':>20} {'€' + f'{rf_rmse:,.0f}':>20}")
print(f"  {'MAE':<8} {'€' + f'{lr_mae:,.0f}':>20} {'€' + f'{rf_mae:,.0f}':>20}")
print(f"  {'R²':<8} {lr_r2:>20.4f} {rf_r2:>20.4f}")

print("\n[7] Plotting feature importance...")

importance_df = pd.DataFrame({
    "Feature"   : FEATURES,
    "Importance": rf_model.feature_importances_
}).sort_values("Importance", ascending=False)

plt.figure(figsize=(9, 5))
sns.barplot(
    data      = importance_df,
    x         = "Importance",
    y         = "Feature",
    palette   = "Blues_r"
)
plt.title("Random Forest — Feature Importance")
plt.xlabel("Relative Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("plots/chart6_feature_importance.png", dpi=150)
print("    Saved: plots/chart6_feature_importance.png")
plt.show()
plt.close()

print("\n[8] Plotting actual vs predicted prices...")

sample_idx    = np.random.choice(len(y_test), size=2000, replace=False)
y_test_sample = np.array(y_test)[sample_idx]
rf_pred_sample = rf_preds[sample_idx]

plt.figure(figsize=(8, 7))
plt.scatter(y_test_sample, rf_pred_sample, alpha=0.3, s=12, color="#1A3866")
plt.plot([0, 150000], [0, 150000], color="red", linewidth=1.2, label="Perfect prediction")
plt.title("Random Forest — Actual vs Predicted Price")
plt.xlabel("Actual Price (€)")
plt.ylabel("Predicted Price (€)")
plt.legend()
plt.tight_layout()
plt.savefig("plots/chart7_actual_vs_predicted.png", dpi=150)
print("    Saved: plots/chart7_actual_vs_predicted.png")
plt.show()
plt.close()

print("\n[9] Saving model and encoders...")

joblib.dump(rf_model,  "models/model.pkl")
joblib.dump(encoders,  "models/encoders.pkl")
joblib.dump(FEATURES,  "models/features.pkl")

print("    Saved: models/model.pkl")
print("    Saved: models/encoders.pkl")
print("    Saved: models/features.pkl")

print(f"\n{'='*45}")
print(f"  STEP 3 COMPLETE")
print(f"{'='*45}")
print(f"  Best model   : Random Forest")
print(f"  R²           : {rf_r2:.4f}  (explains {rf_r2*100:.1f}% of price variance)")
print(f"  RMSE         : €{rf_rmse:,.0f}  (average prediction error)")
print(f"  MAE          : €{rf_mae:,.0f}  (median prediction error)")
print(f"\n  Top features by importance:")
for _, row in importance_df.iterrows():
    bar = "█" * int(row["Importance"] * 50)
    print(f"    {row['Feature']:<20} {bar}  {row['Importance']:.4f}")
print(f"\n  Next: run python3 step4_api.py")
