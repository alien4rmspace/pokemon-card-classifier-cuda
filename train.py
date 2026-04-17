import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


def print_wrong_paths(model: nn.Module, loader: DataLoader, device: torch.device, tag: str) -> None:
    model.eval()
    sample_index = 0

    with torch.no_grad():
        with torch.cuda.nvtx.range(f"print_wrong_paths:{tag}"):
            for batch_idx, (images, labels) in enumerate(loader):
                with torch.cuda.nvtx.range(f"print_wrong_paths:{tag}:batch_{batch_idx}"):
                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = model(images)
                    preds = torch.argmax(outputs, dim=1)

                    for i in range(len(labels)):
                        if preds[i].item() != labels[i].item():
                            print(loader.dataset.samples[sample_index + i][0])

                    sample_index += len(labels)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    tag: str,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        with torch.cuda.nvtx.range(f"evaluate:{tag}"):
            for batch_idx, (images, labels) in enumerate(loader):
                with torch.cuda.nvtx.range(f"evaluate:{tag}:batch_{batch_idx}"):
                    images = images.to(device)
                    labels = labels.to(device)

                    with torch.cuda.nvtx.range(f"evaluate:{tag}:forward"):
                        outputs = model(images)
                        loss = criterion(outputs, labels)

                    total_loss += loss.item()
                    preds = torch.argmax(outputs, dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy


def main() -> None:
    with torch.cuda.nvtx.range("main"):
        # Device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", device)

        # Paths
        train_dir = "data/pokemon_cards/train"
        val_dir = "data/pokemon_cards/val"
        test_dir = "data/pokemon_cards/test"

        # Training config
        batch_size = 32
        num_epochs = 5
        learning_rate = 0.001

        with torch.cuda.nvtx.range("setup:transforms"):
            # Transforms
            train_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
            ])

            eval_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])

        with torch.cuda.nvtx.range("setup:datasets"):
            # Datasets
            train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
            val_dataset = datasets.ImageFolder(val_dir, transform=eval_transform)
            test_dataset = datasets.ImageFolder(test_dir, transform=eval_transform)

        with torch.cuda.nvtx.range("setup:dataloaders"):
            # Dataloaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=5,pin_memory=True, persistent_workers=True, prefetch_factor=4)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=3,pin_memory=True, persistent_workers=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=3,pin_memory=True, persistent_workers=True)

        # Class info
        class_names = train_dataset.classes
        num_classes = len(class_names)
        print("Classes:", num_classes)

        with torch.cuda.nvtx.range("setup:model"):
            # Model
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            model = model.to(device)

        with torch.cuda.nvtx.range("setup:optimizer"):
            # Loss and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Best metrics
        best_val_accuracy = 0.0
        best_val_loss = float("inf")

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0

            with torch.cuda.nvtx.range(f"train:epoch_{epoch + 1}"):
                for batch_idx, (images, labels) in enumerate(train_loader):
                    with torch.cuda.nvtx.range(f"train:epoch_{epoch + 1}:batch_{batch_idx}"):
                        images = images.to(device, non_blocking=True)
                        labels = labels.to(device, non_blocking=True)

                        with torch.cuda.nvtx.range("train:zero_grad"):
                            optimizer.zero_grad()

                        with torch.cuda.nvtx.range("train:forward"):
                            outputs = model(images)
                            loss = criterion(outputs, labels)

                        with torch.cuda.nvtx.range("train:backward"):
                            loss.backward()

                        with torch.cuda.nvtx.range("train:optimizer_step"):
                            optimizer.step()

                        running_loss += loss.item()

            train_loss = running_loss / len(train_loader)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device, "val")

            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

            saved = False
            if val_acc > best_val_accuracy:
                with torch.cuda.nvtx.range("save:best_accuracy"):
                    best_val_accuracy = val_acc
                    torch.save({
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "class_names": class_names,
                        "num_classes": num_classes,
                        "val_acc": val_acc,
                        "val_loss": val_loss,
                    }, "best_resnet18_pokemon_cards_accuracy.pth")
                    print("Saved best accuracy model")
                    saved = True

            if val_loss < best_val_loss:
                with torch.cuda.nvtx.range("save:best_loss"):
                    best_val_loss = val_loss
                    torch.save({
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "class_names": class_names,
                        "num_classes": num_classes,
                        "val_acc": val_acc,
                        "val_loss": val_loss,
                    }, "best_resnet18_pokemon_cards_loss.pth")
                    print("Saved best loss model")
                    saved = True

            if saved:
                print_wrong_paths(model, val_loader, device, "best")

        # Final test evaluation
        test_loss, test_acc = evaluate(model, test_loader, criterion, device, "test")
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()
