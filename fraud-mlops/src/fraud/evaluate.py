import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve


def choose_threshold(y_true, y_prob, target_precision=0.90):
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    # thr has len-1; align by ignoring last precision/rec
    prec = prec[:-1]
    rec = rec[:-1]
    ok = np.where(prec >= target_precision)[0]
    if len(ok) == 0:
        # fallback: threshold that maximizes F1
        f1 = (2 * prec * rec) / (prec + rec + 1e-12)
        idx = int(np.argmax(f1))
        return float(thr[idx]), float(prec[idx]), float(rec[idx])
    idx = int(ok[np.argmax(rec[ok])])
    return float(thr[idx]), float(prec[idx]), float(rec[idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_parquet", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--out_metrics", required=True)
    ap.add_argument("--out_threshold", required=True)
    args = ap.parse_args()

    test_df = pd.read_parquet(args.test_parquet)
    features = [c for c in test_df.columns if c != "Class"]
    x = test_df[features].values.astype("float32")
    y = test_df["Class"].values.astype("int32")

    ckpt = torch.load(args.model_path, map_location="cpu")
    model_features = ckpt["features"]
    if model_features != features:
        raise ValueError("Feature mismatch between model and test data.")

    hidden = ckpt["hidden"]
    dropout = ckpt["dropout"]

    # rebuild the same MLP
    import torch.nn as nn

    class MLP(nn.Module):
        def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(1)

    model = MLP(len(features), hidden=hidden, dropout=dropout)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    with torch.no_grad():
        logits = model(torch.from_numpy(x))
        prob = torch.sigmoid(logits).numpy()

    auprc = float(average_precision_score(y, prob))
    roc = float(roc_auc_score(y, prob))

    thr, prec, rec = choose_threshold(y, prob, target_precision=0.90)

    metrics = {
        "test_auprc": auprc,
        "test_roc_auc": roc,
        "threshold_selection": {"target_precision": 0.90, "threshold": thr, "precision": prec, "recall": rec},
    }

    Path(args.out_metrics).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    Path(args.out_threshold).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_threshold, "w", encoding="utf-8") as f:
        json.dump({"threshold": thr}, f, indent=2)

    print("Evaluation done.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
