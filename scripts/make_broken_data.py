"""Create a deliberately broken CSV to demonstrate a FAILED Validate execution.

    python scripts/make_broken_data.py
    python -m src.cli run --data data/broken_job_postings.csv --name broken-demo

Breaks three validation rules at once: a required column is missing, some
labels are invalid, and many rows have empty text.
"""
import argparse

import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--input", default="data/fake_job_postings.csv")
p.add_argument("--output", default="data/broken_job_postings.csv")
a = p.parse_args()

df = pd.read_csv(a.input).drop(columns=["company_profile"])
df["fraudulent"] = df["fraudulent"].astype(object)
df.loc[df.sample(frac=0.03, random_state=1).index, "fraudulent"] = "unknown"
blank = df.sample(frac=0.25, random_state=2).index
df.loc[blank, ["title", "description", "requirements", "benefits"]] = ""
df.to_csv(a.output, index=False)
print(f"wrote {a.output}: {len(df)} rows, no 'company_profile' column, "
      f"3% invalid labels, 25% empty text")
