import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn


# First load the ResNet-18 classifier
resnet18 = torchvision.models.resnet18(weights=None, progress=True)

# Reduce kernel size and stride as CIFAR10 images are very small
resnet18.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

# Remove max pooling layer for same reason
resnet18.maxpool = nn.Identity()

# Remove final classification layer (as embeddings are in penultimate layer)
resnet18.fc = nn.Identity()

# Define projection head (only used during training)
class ProjectionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 512)
        self.bn = nn.BatchNorm1d(512)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(512, 128)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.fc2(x)
        return F.normalize(x, dim=1)

# Define SimCLR model (a wrapper of the ResNet-18 and projection head)
class SimCLR(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = resnet18
        self.projection = ProjectionHead()

    def forward(self, x1, x2):
        h1 = self.encoder(x1)  # (batch_size, 512)
        h2 = self.encoder(x2)  # (batch_size, 512)

        z1 = self.projection(h1)  # (batch_size, 128)
        z2 = self.projection(h2)  # (batch_size, 128)

        return z1, z2

    # Use after training for embedding extraction
    def get_embedding(self, x):
        return self.encoder(x)