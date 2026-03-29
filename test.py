import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load("pokemon_classification_model_v1_94.pth", map_location=device)
print(checkpoint.keys())

num_classes = checkpoint["num_classes"]
class_names = checkpoint["class_names"]

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

# Compare second model
checkpoint = torch.load("pokemon_classification_model_v1_98.pth", map_location=device)
model_compare = models.resnet18(weights=None)
model_compare.fc = nn.Linear(model_compare.fc.in_features, num_classes)
model_compare.load_state_dict(checkpoint["model_state_dict"])
model_compare.to(device)
model_compare.eval()

# Testing
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

image_path = "data/pokemon_cards/test/charizard/Charizard.png"
image = Image.open(image_path).convert("RGB")
image_tensor = eval_transform(image).unsqueeze(0).to(device)

model.eval()
with torch.no_grad():
    outputs = model(image_tensor)
    _, pred = torch.max(outputs, 1)

    outputs_2 = model_compare(image_tensor)
    _, pred_2 = torch.max(outputs, 1)

predicted_class = class_names[pred.item()]
print(f"Predicted class: {predicted_class}")

predicted_class = class_names[pred_2.item()]
print(f"Predicted class for model 2: {predicted_class}")

if __name__ == "__main__":
    checkpoint = torch.load("pokemon_classification_model_v1_94.pth")
    print(
        "{\n"
        f"  'class_names': {checkpoint['class_names']},\n"
        f"  'num_classes': {checkpoint['num_classes']},\n"
        f"  'val_acc': {checkpoint['val_acc']}\n"
        "}"
    )