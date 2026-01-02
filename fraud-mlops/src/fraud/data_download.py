import argparse
import shutil
from pathlib import Path

import kagglehub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mlg-ulb/creditcardfraud")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    path = kagglehub.dataset_download(args.dataset)
    src = Path(path) / "creditcard.csv"
    if not src.exists():
        raise FileNotFoundError(f"Expected {src} not found. Downloaded path: {path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, out_dir / "creditcard.csv")
    print(f"Saved to: {out_dir / 'creditcard.csv'}")


if __name__ == "__main__":
    main()