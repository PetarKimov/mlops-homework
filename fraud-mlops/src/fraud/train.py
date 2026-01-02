import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import average_precision_score, roc_auc_score


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


@torch.no_grad()
def eval_metrics(model, loader, device):
    model.eval()
    ys = []
    ps = []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        probs = torch.sigmoid(logits).cpu().numpy()
        ys.append(yb.numpy())
        ps.append(probs)
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    auprc = float(average_precision_score(y, p))
    roc = float(roc_auc_score(y, p))
    return auprc, roc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_parquet", required=True)
    ap.add_argument("--val_parquet", required=True)
    ap.add_argument("--out_model", required=True)
    ap.add_argument("--out_metrics", required=True)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_df = pd.read_parquet(args.train_parquet)
    val_df = pd.read_parquet(args.val_parquet)

    features = [c for c in train_df.columns if c != "Class"]
    x_train = train_df[features].values.astype("float32")
    y_train = train_df["Class"].values.astype("float32")
    x_val = val_df[features].values.astype("float32")
    y_val = val_df["Class"].values.astype("float32")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = MLP(in_dim=len(features), hidden=args.hidden, dropout=args.dropout).to(device)

    # pos_weight for imbalance
    n_pos = max(1.0, float(y_train.sum()))
    n_neg = float(len(y_train) - y_train.sum())
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_auprc = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        val_auprc, val_roc = eval_metrics(model, val_loader, device)
        train_loss = float(np.mean(losses))

        # IMPORTANT: print metric in a Katib-friendly line
        print(f"Epoch {epoch:02d} train_loss={train_loss:.6f} val_auprc={val_auprc:.6f} val_roc_auc={val_roc:.6f}")

        if val_auprc > best_auprc:
            best_auprc = val_auprc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = model.state_dict()

    Path(args.out_model).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "features": features,
            "hidden": args.hidden,
            "dropout": args.dropout,
        },
        args.out_model,
    )

    metrics = {
        "best_val_auprc": float(best_auprc),
        "pos_weight": float((n_neg / n_pos)),
        "device": device,
        "hyperparams": {
            "hidden": args.hidden,
            "dropout": args.dropout,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
        },
    }
    Path(args.out_metrics).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Training done.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()