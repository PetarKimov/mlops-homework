import os
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import kagglehub

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
)

# -----------------------------
# Reproducibility
# -----------------------------
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------
# Config
# -----------------------------
@dataclass
class Config:
    seed: int = 42
    test_size: float = 0.20
    val_size: float = 0.20  # fraction of train+val used for val
    batch_size: int = 4096
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 10
    hidden1: int = 128
    hidden2: int = 64
    dropout: float = 0.2
    threshold: float = 0.5  # for confusion matrix / classification report
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model_out: str = "fraud_mlp.pt"
    artifacts_dir: str = "artifacts"


# -----------------------------
# Model
# -----------------------------
class MLP(nn.Module):
    def __init__(self, in_dim: int, h1: int, h2: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, 1),  # logits
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


# -----------------------------
# Data loading
# -----------------------------
def find_creditcard_csv(root_dir: str) -> str:
    # kagglehub usually returns a directory path; the csv is typically creditcard.csv
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.lower() == "creditcard.csv":
                return os.path.join(dirpath, f)
    raise FileNotFoundError(f"Could not find creditcard.csv under {root_dir}")


def load_data() -> pd.DataFrame:
    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    print("Path to dataset files:", path)
    csv_path = find_creditcard_csv(path)
    print("Using:", csv_path)
    df = pd.read_csv(csv_path)
    return df


# -----------------------------
# Train / eval helpers
# -----------------------------
@torch.no_grad()
def predict_proba(model: nn.Module, loader: DataLoader, device: str) -> np.ndarray:
    model.eval()
    probs = []
    for xb, _ in loader:
        xb = xb.to(device)
        logits = model(xb)
        p = torch.sigmoid(logits).detach().cpu().numpy()
        probs.append(p)
    return np.concatenate(probs, axis=0)


def run_epoch(model, loader, optimizer, criterion, device: str) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        bs = xb.size(0)
        total_loss += loss.item() * bs
        n += bs
    return total_loss / max(n, 1)


def main():
    cfg = Config()
    seed_everything(cfg.seed)
    print("Device:", cfg.device)

    df = load_data()

    # Features/label
    y = df["Class"].astype(np.float32).values
    X = df.drop(columns=["Class"]).astype(np.float32).values

    # Stratified split: train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.seed, stratify=y
    )

    # Split train vs val from trainval
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=cfg.val_size,
        random_state=cfg.seed,
        stratify=y_trainval,
    )

    # Scale features (safe + common; V1..V28 already PCA but scaling doesn't hurt)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    # Build TensorDatasets / Loaders
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    # Imbalance handling:
    # BCEWithLogitsLoss with pos_weight = (#neg / #pos) makes positive errors "costlier".
    n_pos = float((y_train == 1).sum())
    n_neg = float((y_train == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=cfg.device)
    print(f"Train positives: {int(n_pos)} / {len(y_train)}  => pos_weight={pos_weight.item():.2f}")

    model = MLP(in_dim=X_train.shape[1], h1=cfg.hidden1, h2=cfg.hidden2, dropout=cfg.dropout).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_auprc = -1.0
    best_state = None

    for epoch in range(1, cfg.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, criterion, cfg.device)

        # Validation metrics
        val_probs = predict_proba(model, val_loader, cfg.device)
        val_auprc = average_precision_score(y_val, val_probs)
        val_rocauc = roc_auc_score(y_val, val_probs)

        print(
            f"Epoch {epoch:02d}/{cfg.epochs} | "
            f"train_loss={train_loss:.6f} | val_AUPRC={val_auprc:.6f} | val_ROC-AUC={val_rocauc:.6f}"
        )

        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            best_state = {
                "model": model.state_dict(),
                "scaler_mean": scaler.mean_,
                "scaler_scale": scaler.scale_,
                "config": cfg.__dict__,
            }

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state["model"])

    # Test metrics
    test_probs = predict_proba(model, test_loader, cfg.device)
    test_auprc = average_precision_score(y_test, test_probs)
    test_rocauc = roc_auc_score(y_test, test_probs)
    print("\n=== Test Metrics ===")
    print(f"AUPRC:   {test_auprc:.6f}")
    print(f"ROC-AUC: {test_rocauc:.6f}")
    prec_t, rec_t, thr_t = precision_recall_curve(y_test, test_probs)
    f1_t = 2 * (prec_t * rec_t) / (prec_t + rec_t + 1e-12)
    best_idx_t = int(np.nanargmax(f1_t))
    best_thr_t = thr_t[best_idx_t - 1] if best_idx_t > 0 and best_idx_t - 1 < len(thr_t) else 0.5
    print(f"Best test F1 threshold (for inspection only): ~{best_thr_t:.4f}")

    # Threshold-based report (not the main metric, but still useful)
    y_pred = (test_probs >= cfg.threshold).astype(np.int32)
    cm = confusion_matrix(y_test.astype(np.int32), y_pred)
    print(f"\nConfusion matrix @ threshold={cfg.threshold}:")
    print(cm)
    print("\nClassification report:")
    print(classification_report(y_test.astype(np.int32), y_pred, digits=4))

    # Optional: find threshold that maximizes F1 on validation (handy later for serving)
    prec, rec, thr = precision_recall_curve(y_val, val_probs)
    f1 = 2 * (prec * rec) / (prec + rec + 1e-12)
    best_idx = int(np.nanargmax(f1))
    best_thr = thr[best_idx - 1] if best_idx > 0 and best_idx - 1 < len(thr) else 0.5
    print(f"Suggested threshold (max F1 on val): ~{best_thr:.4f}")

    # -----------------------------
    # Save artifacts
    # -----------------------------
    os.makedirs(cfg.artifacts_dir, exist_ok=True)

    model_path = os.path.join(cfg.artifacts_dir, cfg.model_out)
    torch.save(best_state, model_path)

    np.save(os.path.join(cfg.artifacts_dir, "scaler_mean.npy"), scaler.mean_)
    np.save(os.path.join(cfg.artifacts_dir, "scaler_scale.npy"), scaler.scale_)

    print("\nSaved artifacts:")
    print(" -", model_path)
    print(" -", os.path.join(cfg.artifacts_dir, "scaler_mean.npy"))
    print(" -", os.path.join(cfg.artifacts_dir, "scaler_scale.npy"))


if __name__ == "__main__":
    main()