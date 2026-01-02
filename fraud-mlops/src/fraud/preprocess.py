import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


FEATURES = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
LABEL = "Class"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--out_train", required=True)
    ap.add_argument("--out_val", required=True)
    ap.add_argument("--out_test", required=True)
    ap.add_argument("--out_scaler", required=True)
    ap.add_argument("--out_schema", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)
    # Basic sanity
    missing = [c for c in FEATURES + [LABEL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Split stratified
    y = df[LABEL].astype(int)
    X = df[FEATURES]

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=args.seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=args.seed, stratify=y_tmp
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Save parquet splits (features + label)
    train_out = pd.DataFrame(X_train_scaled, columns=FEATURES)
    train_out[LABEL] = y_train.values
    val_out = pd.DataFrame(X_val_scaled, columns=FEATURES)
    val_out[LABEL] = y_val.values
    test_out = pd.DataFrame(X_test_scaled, columns=FEATURES)
    test_out[LABEL] = y_test.values

    Path(args.out_train).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_val).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_test).parent.mkdir(parents=True, exist_ok=True)

    train_out.to_parquet(args.out_train, index=False)
    val_out.to_parquet(args.out_val, index=False)
    test_out.to_parquet(args.out_test, index=False)

    Path(args.out_scaler).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, args.out_scaler)

    schema = {
        "features": FEATURES,
        "label": LABEL,
        "n_rows": int(len(df)),
        "pos_rate": float(y.mean()),
        "splits": {
            "train": int(len(train_out)),
            "val": int(len(val_out)),
            "test": int(len(test_out)),
        },
    }
    Path(args.out_schema).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_schema, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    print("Preprocess done.")
    print(json.dumps(schema, indent=2))


if __name__ == "__main__":
    main()