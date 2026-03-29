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
    with torch.cuda.nvtx.range(f"load_model:{checkpoint_path}"):
        checkpoint = torch.load(checkpoint_path, map_location=device)

        class_names: list[str] = checkpoint["class_names"]
        num_classes: int = checkpoint["num_classes"]

        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

    return model, class_names, num_classes


def build_test_dataset(test_dir: str) -> datasets.ImageFolder:
    with torch.cuda.nvtx.range("build_test_dataset"):
        eval_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        return datasets.ImageFolder(test_dir, transform=eval_transform)


def build_loader(dataset: datasets.ImageFolder, batch_size: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device, model_name: str) -> float:
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        with torch.cuda.nvtx.range(f"evaluate:{model_name}"):
            for batch_idx, (images, labels) in enumerate(loader):
                with torch.cuda.nvtx.range(f"{model_name}:batch_{batch_idx}"):
                    images = images.to(device)
                    labels = labels.to(device)

                    with torch.cuda.nvtx.range(f"{model_name}:forward"):
                        outputs = model(images)
                        preds = outputs.argmax(dim=1)

                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

    return correct / total if total > 0 else 0.0


def print_sample_predictions(
    model_1: nn.Module,
    model_2: nn.Module,
    loader: DataLoader,
    dataset: datasets.ImageFolder,
    dataset_class_names: list[str],
    checkpoint_class_names: list[str],
    device: torch.device,
) -> None:
    model_1.eval()
    model_2.eval()

    sample_offset = 0

    with torch.no_grad():
        with torch.cuda.nvtx.range("print_sample_predictions"):
            for images, labels in loader:
                images = images.to(device)

                with torch.cuda.nvtx.range("sample:model_1_forward"):
                    outputs_1 = model_1(images)
                    preds_1 = outputs_1.argmax(dim=1)

                with torch.cuda.nvtx.range("sample:model_2_forward"):
                    outputs_2 = model_2(images)
                    preds_2 = outputs_2.argmax(dim=1)

                for i in range(len(labels)):
                    image_path = dataset.samples[sample_offset + i][0]
                    true_label = dataset_class_names[labels[i].item()]
                    pred_1_label = checkpoint_class_names[preds_1[i].item()]
                    pred_2_label = checkpoint_class_names[preds_2[i].item()]

                    print(
                        f"Path: {image_path} | "
                        f"True: {true_label} | "
                        f"Model 1: {pred_1_label} | "
                        f"Model 2: {pred_2_label}"
                    )

                sample_offset += len(labels)


def validate_class_mapping(
    dataset_classes: list[str],
    checkpoint_classes_1: list[str],
    checkpoint_classes_2: list[str],
    num_classes_1: int,
    num_classes_2: int,
) -> None:
    print("Test dataset classes:", dataset_classes)
    print("Checkpoint 1 classes:", checkpoint_classes_1)
    print("Checkpoint 2 classes:", checkpoint_classes_2)

    if checkpoint_classes_1 != checkpoint_classes_2 or num_classes_1 != num_classes_2:
        raise ValueError("The two checkpoints do not use the same class mapping.")

    if dataset_classes != checkpoint_classes_1:
        print("\nWARNING: test_dataset.classes does not match checkpoint class_names.")
        print("This will make accuracy incorrect unless the class order is identical.\n")


def main() -> None:
    with torch.cuda.nvtx.range("main"):
        test_dataset = build_test_dataset(TEST_DIR)
        test_loader = build_loader(test_dataset, BATCH_SIZE)

        print("Number of classes:", len(test_dataset.classes))
        print("Number of test images:", len(test_dataset))

        model_1, class_names_1, num_classes_1 = load_model(MODEL_PATH_1, device)
        model_2, class_names_2, num_classes_2 = load_model(MODEL_PATH_2, device)

        validate_class_mapping(
            test_dataset.classes,
            class_names_1,
            class_names_2,
            num_classes_1,
            num_classes_2,
        )

        acc_1 = evaluate_model(model_1, test_loader, device, "model_1")
        acc_2 = evaluate_model(model_2, test_loader, device, "model_2")

        print(f"Model 1 Test Accuracy: {acc_1:.4f}")
        print(f"Model 2 Test Accuracy: {acc_2:.4f}")

        print("\nSample predictions from first test batch:")
        print_sample_predictions(
            model_1,
            model_2,
            test_loader,
            test_dataset,
            test_dataset.classes,
            class_names_1,
            device,
        )


if __name__ == "__main__":
    main()