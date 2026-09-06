from io import StringIO
import pandas as pd

print("Loading raw dataset...")

with open("data/autos.csv", "r", encoding="utf-8", errors="replace") as f:
    raw = f.read()

raw = raw.replace("\u2028", " ").replace("\u2029", " ")
df  = pd.read_csv(StringIO(raw))

rows_start = len(df)
print(f"  Loaded {rows_start:,} rows and {df.shape[1]} columns")

df["price_in_euro"] = pd.to_numeric(df["price_in_euro"], errors="coerce")
df["year"]          = pd.to_numeric(df["year"],          errors="coerce")
df["power_kw"]      = pd.to_numeric(df["power_kw"],      errors="coerce")

print("\n[1] Removing column-shifted rows...")

VALID_FUEL_TYPES = ["Petrol", "Diesel", "Hybrid", "Electric", "LPG", "CNG", "Others"]
before = len(df)
df     = df[df["fuel_type"].isin(VALID_FUEL_TYPES)]
print(f"    Removed {before - len(df):,} bad rows → {len(df):,} remaining")

print("\n[2] Filtering price outliers (keep €500 – €150,000)...")
before = len(df)
df     = df[df["price_in_euro"].between(500, 150_000)]
print(f"    Removed {before - len(df):,} rows → {len(df):,} remaining")

print("\n[3] Filtering mileage outliers (keep 0 – 500,000 km)...")
before = len(df)
df     = df[df["mileage_in_km"].between(0, 500_000)]
print(f"    Removed {before - len(df):,} rows → {len(df):,} remaining")

print("\n[4] Filtering year outliers (keep 1990 – 2024)...")
before = len(df)
df     = df[df["year"].between(1990, 2024)]
print(f"    Removed {before - len(df):,} rows → {len(df):,} remaining")

print("\n[5] Filtering power outliers (keep 10 – 1,000 kW)...")
before = len(df)
df     = df[df["power_kw"].between(10, 1_000)]
print(f"    Removed {before - len(df):,} rows → {len(df):,} remaining")

print("\n[6] Dropping unnecessary columns...")

cols_to_drop = [
    "Unnamed: 0",
    "registration_date",
    "power_ps",
    "fuel_consumption_l_100km",
    "fuel_consumption_g_km",
    "offer_description",
]

df = df.drop(columns=cols_to_drop)
print(f"    Dropped: {cols_to_drop}")
print(f"    Remaining columns: {list(df.columns)}")

print("\n[7] Dropping rows with missing values in key columns...")

key_cols = [
    "price_in_euro", "year", "mileage_in_km",
    "power_kw", "brand", "fuel_type", "transmission_type", "color"
]
before = len(df)
df     = df.dropna(subset=key_cols)
print(f"    Removed {before - len(df):,} rows → {len(df):,} remaining")

print("\n[8] Filtering transmission types...")
VALID_TRANSMISSION = ["Automatic", "Manual", "Semi-automatic"]
before = len(df)
df     = df[df["transmission_type"].isin(VALID_TRANSMISSION)]
print(f"    Removed {before - len(df):,} rows → {len(df):,} remaining")

df["year"] = df["year"].astype(int)

df = df.reset_index(drop=True)

df.to_csv("data/autos_clean.csv", index=False)

rows_end = len(df)

print(f"\n{'='*45}")
print(f"  STEP 2 COMPLETE")
print(f"{'='*45}")
print(f"  Started with  : {rows_start:,} rows")
print(f"  Cleaned to    : {rows_end:,} rows")
print(f"  Removed total : {rows_start - rows_end:,} rows  ({(rows_start - rows_end) / rows_start * 100:.1f}%)")
print(f"\n  Final columns ({df.shape[1]}):")
for col in df.columns:
    print(f"    - {col}  [{df[col].dtype}]")
print(f"\n  Price range (clean)  : €{df['price_in_euro'].min():,.0f} – €{df['price_in_euro'].max():,.0f}")
print(f"  Median price (clean) : €{df['price_in_euro'].median():,.0f}")
print(f"  Mileage range (clean): {df['mileage_in_km'].min():,.0f} – {df['mileage_in_km'].max():,.0f} km")
print(f"  Year range (clean)   : {df['year'].min()} – {df['year'].max()}")
print(f"\n  Preview of clean data:")
print(df.head())
print(f"\n  Saved to: data/autos_clean.csv")
print(f"\n  Next: run python3 step3_train.py")
