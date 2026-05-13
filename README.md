"""
train.py
CNN modelini eğitir.

Kullanım (lokal veya Colab):
    python train.py --sessions surus_01 surus_02 --arch gtanet --epochs 30

Colab için:
    !python train.py --sessions surus_01 --arch gtanet --epochs 50 --batch 64
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Proje modülleri
import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.cnn_model import build_model, save_model, count_parameters
from models.dataset   import build_loaders


# ─────────────────────────────────────────────────────────────
# Eğitim döngüsü
# ─────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for frames, labels in loader:
        frames = frames.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(frames)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * frames.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += frames.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for frames, labels in loader:
        frames = frames.to(device)
        labels = labels.to(device)

        logits = model(frames)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * frames.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += frames.size(0)

    return total_loss / total, correct / total


# ─────────────────────────────────────────────────────────────
# Ana eğitim fonksiyonu
# ─────────────────────────────────────────────────────────────
def train(
    sessions:    list[str],
    arch:        str   = "gtanet",
    epochs:      int   = 30,
    batch_size:  int   = 32,
    lr:          float = 1e-3,
    val_split:   float = 0.2,
    save_dir:    str   = "checkpoints",
    num_workers: int   = 2,
):
    # ── Cihaz ──
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"[Train] Cihaz: {device}")

    # ── Train / Val bölme ──
    split_idx   = max(1, int(len(sessions) * (1 - val_split)))
    train_sess  = sessions[:split_idx]
    val_sess    = sessions[split_idx:] if split_idx < len(sessions) else sessions[:1]
    print(f"[Train] Train oturumları: {train_sess}")
    print(f"[Train] Val oturumları  : {val_sess}")

    # ── DataLoader'lar ──
    train_loader, val_loader = build_loaders(
        train_sess, val_sess,
        batch_size=batch_size, num_workers=num_workers
    )

    # ── Model ──
    model = build_model(arch).to(device)
    print(f"[Train] Model: {arch.upper()}  |  Parametre: {count_parameters(model):,}")

    # ── Loss, Optimizer, Scheduler ──
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5, verbose=True)

    # ── Checkpoint klasörü ──
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # ── Eğitim ──
    history    = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val   = float("inf")
    best_epoch = 0

    print(f"\n{'─'*60}")
    print(f"{'Epoch':>6} {'TrLoss':>8} {'TrAcc':>7} {'VaLoss':>8} {'VaAcc':>7} {'LR':>9}")
    print(f"{'─'*60}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = validate(model, val_loader, criterion, device)

        scheduler.step(va_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        elapsed = time.time() - t0
        print(f"{epoch:>6} {tr_loss:>8.4f} {tr_acc:>6.2%} {va_loss:>8.4f} {va_acc:>6.2%} {current_lr:>9.2e}  ({elapsed:.1f}s)")

        # En iyi modeli kaydet
        if va_loss < best_val:
            best_val   = va_loss
            best_epoch = epoch
            save_model(model, save_path / f"best_{arch}.pth")
            print(f"          ★ Yeni en iyi model kaydedildi (epoch {epoch})")

    # Son modeli kaydet
    save_model(model, save_path / f"last_{arch}.pth")

    # History JSON
    with open(save_path / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'─'*60}")
    print(f"  Eğitim tamamlandı!")
    print(f"  En iyi val loss: {best_val:.4f}  (epoch {best_epoch})")
    print(f"  Model kayıt yeri: {save_path}/")
    print(f"{'─'*60}\n")

    return history


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GTA SA CNN Eğitimi")
    parser.add_argument("--sessions",    nargs="+", required=True,
                        help="Oturum adları: surus_01 surus_02 ...")
    parser.add_argument("--arch",        default="gtanet",
                        choices=["gtanet", "resnet"])
    parser.add_argument("--epochs",      type=int,   default=30)
    parser.add_argument("--batch",       type=int,   default=32)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--val_split",   type=float, default=0.2)
    parser.add_argument("--save_dir",    default="checkpoints")
    parser.add_argument("--workers",     type=int,   default=2)

    args = parser.parse_args()

    train(
        sessions    = args.sessions,
        arch        = args.arch,
        epochs      = args.epochs,
        batch_size  = args.batch,
        lr          = args.lr,
        val_split   = args.val_split,
        save_dir    = args.save_dir,
        num_workers = args.workers,
    )
