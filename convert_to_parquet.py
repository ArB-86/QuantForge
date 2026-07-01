import pandas as pd

CSV = "../data/training/master_v9.csv"          # <-- replace with your actual CSV path
PARQUET = "../data/training/master_v9.parquet"

print("Loading CSV...")
df = pd.read_csv(CSV)

print("Writing Parquet...")
df.to_parquet(
    PARQUET,
    index=False,
)

print("Done.")
