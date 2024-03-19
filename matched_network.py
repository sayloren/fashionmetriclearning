import torch.nn as nn

# Matched Network Model
# regularizaation and normalization....architecture changes
class MatchedNetwork(nn.Module):
    def __init__(self):
        super(MatchedNetwork, self).__init__()
        
        # Define the convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Define the fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 3 * 3, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128))

    def forward_once(self, x):
        # Forward pass through the convolutional layers
        x = self.conv_layers(x)
        # Reshape the output for the fully connected layers
        x = x.view(x.size(0), -1)
        # Forward pass through the fully connected layers
        x = self.fc_layers(x)
        return x

    def forward(self, input1, input2, input3):
        # Forward pass through each branch of the network
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output3 = self.forward_once(input3)
        return output1, output2, output3
        