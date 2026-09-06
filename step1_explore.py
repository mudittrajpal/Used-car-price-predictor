import os
from io import StringIO

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

output_folder = "plots"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 11, "axes.titlesize": 13})

print("Loading dataset...")

with open("data/autos.csv", "r", encoding="utf-8", errors="replace") as f:
    raw = f.read()

raw = raw.replace("\u2028", " ").replace("\u2029", " ")
df  = pd.read_csv(StringIO(raw))

df["price_in_euro"] = pd.to_numeric(df["price_in_euro"], errors="coerce")
df["year"]          = pd.to_numeric(df["year"],          errors="coerce")
df["power_kw"]      = pd.to_numeric(df["power_kw"],      errors="coerce")
df["power_ps"]      = pd.to_numeric(df["power_ps"],      errors="coerce")

print("\n=============================================")
print("  STEP 1 — GETTING TO KNOW THE DATASET")
print("=============================================")

print(f"\n  Rows   : {df.shape[0]:,}")
print(f"  Columns: {df.shape[1]}")

print("\n--- Column names and data types ---")
print(df.dtypes)

print("\n--- First 5 rows ---")
print(df.head())

print("\n--- Missing values (NaN only) ---")
missing     = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
report      = pd.DataFrame({"Count": missing, "%": missing_pct})
report      = report[report["Count"] > 0].sort_values("%", ascending=False)

if report.empty:
    print("  No NaN values found.")
else:
    print(report)

print("\n  Note: fuel_consumption_l_100km also uses '- (g/km)' as a")
print("  placeholder for missing data — cleaned in step2_clean.py.")

print("\n--- Descriptive statistics (numeric columns) ---")
print(df[["price_in_euro", "year", "power_kw", "power_ps", "mileage_in_km"]].describe())

print("\n--- Top 10 brands by listing count ---")
print(df["brand"].value_counts().head(10))

print("\n--- Transmission types ---")
print(df["transmission_type"].value_counts())

print("\n--- Fuel types (top 8 — rest are junk from shifted rows) ---")
print(df["fuel_type"].value_counts().head(8))

print("\n--- Top 10 colours ---")
print(df["color"].value_counts().head(10))

print(f"\n--- Year range ---")
print(f"  Oldest : {int(df['year'].min())}")
print(f"  Newest : {int(df['year'].max())}")
print(f"  Most common: {int(df['year'].mode()[0])}")

print("\n--- Price outliers (raw, before cleaning) ---")
print(f"  Min    : €{df['price_in_euro'].min():,.0f}")
print(f"  Max    : €{df['price_in_euro'].max():,.0f}")
print(f"  Median : €{df['price_in_euro'].median():,.0f}")
print(f"\n  Listings priced at €0         : {(df['price_in_euro'] == 0).sum():,}")
print(f"  Listings priced over €150,000 : {(df['price_in_euro'] > 150000).sum():,}")

print("\n--- Mileage outliers (raw, before cleaning) ---")
print(f"  Min : {df['mileage_in_km'].min():,.0f} km")
print(f"  Max : {df['mileage_in_km'].max():,.0f} km")

print("\nGenerating charts...")

plt.figure(figsize=(10, 5))
price_capped = df[df["price_in_euro"].between(500, 80000)]["price_in_euro"]
plt.hist(price_capped, bins=60, color="#1A3866", edgecolor="white", linewidth=0.4)
plt.title("Distribution of Car Prices (€500 – €80,000)")
plt.xlabel("Price (€)")
plt.ylabel("Number of Listings")
plt.tight_layout()
plt.savefig(output_folder + "/chart1_price_distribution.png", dpi=150)
print("  Saved: chart1_price_distribution.png")
plt.show()
plt.close()

plt.figure(figsize=(10, 5))
mileage_capped = df[df["mileage_in_km"].between(0, 500000)]["mileage_in_km"]
plt.hist(mileage_capped, bins=60, color="#2563EB", edgecolor="white", linewidth=0.4)
plt.title("Distribution of Mileage (up to 500,000 km)")
plt.xlabel("Mileage (km)")
plt.ylabel("Number of Listings")
plt.tight_layout()
plt.savefig(output_folder + "/chart2_mileage_distribution.png", dpi=150)
print("  Saved: chart2_mileage_distribution.png")
plt.show()
plt.close()

plt.figure(figsize=(12, 5))
year_counts = df["year"].value_counts().sort_index()
year_counts = year_counts[(year_counts.index >= 1990) & (year_counts.index <= 2024)]
plt.bar(year_counts.index, year_counts.values, color="#1A3866", edgecolor="white", linewidth=0.3)
plt.title("Number of Listings by Registration Year (1990–2024)")
plt.xlabel("Year")
plt.ylabel("Number of Listings")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(output_folder + "/chart3_listings_by_year.png", dpi=150)
print("  Saved: chart3_listings_by_year.png")
plt.show()
plt.close()

plt.figure(figsize=(12, 5))
top_brands = df["brand"].value_counts().head(15)
sns.barplot(x=top_brands.index, y=top_brands.values, palette="Blues_r")
plt.title("Top 15 Brands by Number of Listings")
plt.xlabel("Brand")
plt.ylabel("Number of Listings")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(output_folder + "/chart4_top_brands.png", dpi=150)
print("  Saved: chart4_top_brands.png")
plt.show()
plt.close()

plt.figure(figsize=(9, 5))
valid_fuels    = df["fuel_type"].value_counts()
valid_fuels    = valid_fuels[valid_fuels > 100].index
df_fuel        = df[df["fuel_type"].isin(valid_fuels)]
median_by_fuel = df_fuel.groupby("fuel_type")["price_in_euro"].median().sort_values(ascending=False)
sns.barplot(x=median_by_fuel.index, y=median_by_fuel.values, palette="Blues_r")
plt.title("Median Listing Price by Fuel Type")
plt.xlabel("Fuel Type")
plt.ylabel("Median Price (€)")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(output_folder + "/chart5_price_by_fuel.png", dpi=150)
print("  Saved: chart5_price_by_fuel.png")
plt.show()
plt.close()

print("\n=============================================")
print("  STEP 1 COMPLETE")
print("=============================================")
print(f"  Total listings   : {len(df):,}")
print(f"  Price range (raw): €{df['price_in_euro'].min():,.0f} – €{df['price_in_euro'].max():,.0f}")
print(f"  Median price     : €{df['price_in_euro'].median():,.0f}")
print(f"  Mileage range    : {df['mileage_in_km'].min():,.0f} – {df['mileage_in_km'].max():,.0f} km")
print(f"  Unique brands    : {df['brand'].nunique()}")
print(f"  Unique models    : {df['model'].nunique()}")
print(f"  Year range       : {int(df['year'].min())} – {int(df['year'].max())}")
print(f"\n  Charts saved to  : {output_folder}/")
print("\n  Next: run python3 step2_clean.py")
