import torch
import torch.nn as nn
import torch.nn.functional as F

class ASClassifier(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(ASClassifier, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, 32)
        # self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x