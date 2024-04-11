import torch
import torch.nn as nn
import torchvision.models as models

class ResNet50CNN(nn.Module):
    def __init__(self, num_classes=5):
        super(ResNet50CNN, self).__init__()
        # Load the pre-trained ResNet50 model
        resnet = models.resnet50(pretrained=True)
        # Remove the original fully connected layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        # Add custom fully connected layers for classification
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, num_classes)
        # Dropout layer
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Extract features using ResNet50
        x = self.features(x)
        # Flatten
        x = x.view(x.size(0), -1)
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Create the model
model = ResNet50CNN()
