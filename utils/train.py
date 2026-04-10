import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train_model(model, train_loader, val_loader, device, epochs=10, log_file=None):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        # -------------------------
        # 1. TRAINING PHASE
        # -------------------------
        model.train()
        running_train_loss = 0.0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        # Correct math: Average loss per batch
        avg_train_loss = running_train_loss / len(train_loader)

        # -------------------------
        # 2. VALIDATION PHASE
        # -------------------------
        model.eval()
        running_val_loss = 0.0
        correct = 0
        total = 0
        
        # Turn off gradients to save VRAM and skip backprop math
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                running_val_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = running_val_loss / len(val_loader)
        val_acc = 100.0 * correct / total

        # -------------------------
        # 3. LOGGING
        # -------------------------
        msg = f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%"
        print(msg)

        if log_file:
            with open(log_file, "a") as f:
                f.write(msg + "\n")

    return model