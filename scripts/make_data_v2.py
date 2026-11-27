"""Create a "v2" of the dataset by dropping a random fraction of rows.

    python scripts/make_data_v2.py --drop-frac 0.10 --seed 7

The new file has a different sha256, so the pipeline registers it as a new
RawDataset version and its lineage branches away from v1.
"""
import argparse

import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--input", default="data/fake_job_postings.csv")
p.add_argument("--output", default="data/fake_job_postings_v2.csv")
p.add_argument("--drop-frac", type=float, default=0.10)
p.add_argument("--seed", type=int, default=7)
a = p.parse_args()

df = pd.read_csv(a.input)
v2 = df.drop(df.sample(frac=a.drop_frac, random_state=a.seed).index)
v2.to_csv(a.output, index=False)
print(f"{a.input}: {len(df)} rows -> {a.output}: {len(v2)} rows "
      f"(dropped {len(df) - len(v2)}, fraud rate {v2['fraudulent'].mean():.2%})")
