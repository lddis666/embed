# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class ASClassifier(nn.Module):
#     def __init__(self, embedding_dim, num_classes):
#         super(ASClassifier, self).__init__()
#         self.fc1 = nn.Linear(embedding_dim, 32)
#         # self.fc2 = nn.Linear(64, 32)
#         self.fc3 = nn.Linear(32, num_classes)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         # x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x




import torch.nn as nn
import torch.nn.functional as F

class ASClassifier(nn.Module):
    def __init__(self, embedding_dim, num_classes, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)      # 可选：对隐层做 BN
        self.dropout = nn.Dropout(dropout)         # 防过拟合
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2) # 可选
        self.out = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.out(x)
        return x