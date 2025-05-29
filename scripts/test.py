import torch
import torch.nn as nn
from torchvision.models import vgg16
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import vgg16


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model architecture
model = vgg16(pretrained=False)  # pretrained=False to avoid overwriting weights
model.classifier[6] = nn.Linear(4096, 10)  # CIFAR-10 has 10 classes
model = model.to(device)

# Load saved state_dict (weights)
model.load_state_dict(torch.load('../model/vgg16_cifar10.pth', map_location='cpu'))

transform = transforms.Compose([
    transforms.Resize(224), 
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 3️⃣ Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False, num_workers=2)


print("Model loaded successfully!")

model.eval()

correct = 0
total = 0

# Disable gradient calculation (inference only)
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy on CIFAR-10 test set: {accuracy:.2f}%')
