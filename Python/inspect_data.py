import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"

# Files to inspect
files = [
    "orders.csv",
    "products.csv",
    "aisles.csv",
    "departments.csv",
    "order_products__train.csv",
    "order_products__prior.csv"
]

for file in files:
    path = RAW_DIR / file

    print("\n" + "=" * 60)
    print(f"FILE: {file}")
    print("=" * 60)

    if not path.exists():
        print("❌ File not found")
        continue

    # Only read a sample for now
    df = pd.read_csv(path, nrows=5)

    print(f"Columns: {list(df.columns)}")
    print(f"Sample rows: {len(df)}")
    print("\nData types:")
    print(df.dtypes)

    print("\nFirst 5 rows:")
    print(df.to_string(index=False))