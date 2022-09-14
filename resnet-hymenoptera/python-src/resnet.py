import torch.nn as nn
import torch.nn.functional as F


class FCnet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FCnet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
