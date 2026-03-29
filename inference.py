import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets

TEST_DIR = "data/pokemon_cards/test"
MODEL_PATH_1 = "pokemon_classification_model_v3_accuracy.pth"
MODEL_PATH_2 = "pokemon_classification_model_v3_loss.pth"
BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path: str, device: torch.device) -> tuple[nn.Module, list[str], int]:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    class_names: list[str] = checkpoint["class_names"]
    num_classes: int = checkpoint["num_classes"]

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, class_names, num_classes


def build_test_loader(test_dir: str, batch_size: int) -> tuple[DataLoader, datasets.ImageFolder]:
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    test_dataset = datasets.ImageFolder(test_dir, transform=eval_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_loader, test_dataset


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total if total > 0 else 0.0


def print_sample_predictions(
    model_1: nn.Module,
    model_2: nn.Module,
    loader: DataLoader,
    dataset_class_names: list[str],
    checkpoint_class_names: list[str],
    device: torch.device,
) -> None:
    model_1.eval()
    model_2.eval()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            outputs_1 = model_1(images)
            preds_1 = outputs_1.argmax(dim=1)

            outputs_2 = model_2(images)
            preds_2 = outputs_2.argmax(dim=1)

            for i in range(len(labels)):
                true_label = dataset_class_names[labels[i].item()]
                pred_1_label = checkpoint_class_names[preds_1[i].item()]
                pred_2_label = checkpoint_class_names[preds_2[i].item()]

                print(f"True: {true_label} | Model 1: {pred_1_label} | Model 2: {pred_2_label}")

            


def main() -> None:
    test_loader, test_dataset = build_test_loader(TEST_DIR, BATCH_SIZE)

    print("Test dataset classes:", test_dataset.classes)
    print("Number of classes:", len(test_dataset.classes))
    print("Number of test images:", len(test_dataset))

    model_1, class_names_1, num_classes_1 = load_model(MODEL_PATH_1, device)
    model_2, class_names_2, num_classes_2 = load_model(MODEL_PATH_2, device)

    print("Checkpoint 1 classes:", class_names_1)
    print("Checkpoint 2 classes:", class_names_2)

    if class_names_1 != class_names_2 or num_classes_1 != num_classes_2:
        raise ValueError("The two checkpoints do not use the same class mapping.")

    if test_dataset.classes != class_names_1:
        print("\nWARNING: test_dataset.classes does not match checkpoint class_names.")
        print("This will make accuracy incorrect unless the class order is identical.\n")

    acc_1 = evaluate_model(model_1, test_loader, device)
    acc_2 = evaluate_model(model_2, test_loader, device)

    print(f"Model 1 Test Accuracy: {acc_1:.4f}")
    print(f"Model 2 Test Accuracy: {acc_2:.4f}")

    print("\nSample predictions from first test batch:")
    print_sample_predictions(
        model_1,
        model_2,
        test_loader,
        test_dataset.classes,
        class_names_1,
        device,
    )


if __name__ == "__main__":
    main()