import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train_model(model, train_loader, val_loader, device, epochs=15, log_file=None):
    """
    Train a single model with:
      - Adam + weight decay
      - Label-smoothed CrossEntropyLoss (reduces overconfidence)
      - ReduceLROnPlateau scheduler (halves LR when val acc plateaus)
      - deep-copied best weights (critical correctness fix)
    Returns the model loaded with the best-validation-accuracy weights.
    """
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=3e-4,
        weight_decay=1e-4,
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, verbose=True
    )

    best_acc = 0.0
    # ✅ deepcopy — stores a real snapshot, not a live reference
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):

        # ── Training ──────────────────────────────────────────
        model.train()
        total_loss  = 0.0
        num_batches = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss  += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)

        # ── Validation ────────────────────────────────────────
        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, pred  = torch.max(outputs, 1)
                total   += labels.size(0)
                correct += (pred == labels).sum().item()

        acc    = 100.0 * correct / max(total, 1)
        lr_now = optimizer.param_groups[0]["lr"]

        msg = (
            f"[{model.__class__.__name__}] "
            f"Epoch {epoch+1}/{epochs}  "
            f"AvgLoss={avg_loss:.4f}  "
            f"ValAcc={acc:.2f}%  "
            f"LR={lr_now:.2e}"
        )
        print(msg)
        if log_file:
            with open(log_file, "a") as lf:
                lf.write(msg + "\n")

        # Step scheduler on validation accuracy
        scheduler.step(acc)

        # ✅ deepcopy — real snapshot of the best epoch
        if acc > best_acc:
            best_acc     = acc
            best_weights = copy.deepcopy(model.state_dict())

    print(f"  ✓  Best val acc for {model.__class__.__name__}: {best_acc:.2f}%\n")
    model.load_state_dict(best_weights)
    return model