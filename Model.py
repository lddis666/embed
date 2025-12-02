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
    

import torch
import torch.nn as nn
import torch.nn.functional as F


# class ToRClassifier(nn.Module):
#     """
#     输入:  x, shape = (batch_size, 2, embedding_dim)
#     输出:  logits, shape = (batch_size, num_classes)
#     """
#     def __init__(self, embedding_dim: int, num_classes: int = 3):
#         super(ToRClassifier, self).__init__()

#         # Conv1d 要求输入 shape = (B, C_in, L)
#         # 其中 C_in 像是通道数, L 像是序列长度
#         # 这里我们把 embedding_dim 当作 C_in, 序列长度为 2
#         # 所以会把 (B, 2, D) 转成 (B, D, 2) 再送入 Conv1d

#         self.conv1 = nn.Conv1d(
#             in_channels=embedding_dim,
#             out_channels=32,
#             kernel_size=3,
#             padding=1   # 对应 Keras 的 padding='same'
#         )

#         self.conv2 = nn.Conv1d(
#             in_channels=32,
#             out_channels=32,
#             kernel_size=3,
#             padding=1
#         )

#         # 池化层
#         self.pool = nn.MaxPool1d(kernel_size=2)

#         # 注意: 原始 Keras 代码里，在长度只有 2 的情况下还做两次池化，
#         # 第二次池化会把长度变成 0，这在 Keras 里其实也不太合理。
#         # 这里给出一个「合理可运行」且结构类似的版本：
#         #
#         #   (1) 先 Conv -> ReLU -> Pool 一次 (长度从2变1)
#         #   (2) 再 Conv -> ReLU (不再 Pool，避免长度为0)
#         #
#         # 如果你一定要完全照抄原始结构，我们可以再讨论如何特殊处理。

#         # 全连接部分的输入维度：
#         # 经过 conv1 + pool 后，长度从 2 -> 1
#         # 经过 conv2 后，仍然是长度 1
#         # 通道数为 32，所以 Flatten 后是 32 * 1 = 32
#         self.fc1 = nn.Linear(32, 100)
#         self.fc2 = nn.Linear(100, num_classes)

#     def forward(self, x):
#         """
#         x: (batch_size, 2, embedding_dim)
#         """
#         # 先换维成 (B, C_in, L) = (B, embedding_dim, 2)
#         x = x.permute(0, 2, 1)   # (B, 2, D) -> (B, D, 2)

#         # 第一次卷积 + ReLU
#         x = self.conv1(x)        # (B, 32, 2)
#         x = F.relu(x)

#         # 第一次池化: 2 -> 1
#         x = self.pool(x)         # (B, 32, 1)

#         # 第二次卷积 + ReLU (不再池化，避免长度归零)
#         x = self.conv2(x)        # (B, 32, 1)
#         x = F.relu(x)

#         # 展平
#         x = x.view(x.size(0), -1)  # (B, 32 * 1) = (B, 32)

#         # 全连接
#         x = F.relu(self.fc1(x))
#         logits = self.fc2(x)       # (B, num_classes)

#         # 训练时一般直接返回 logits，配合 CrossEntropyLoss 使用
#         # 推理时如果要概率可以再套 softmax
#         return logits




class ToRClassifier(nn.Module):
    """
    输入:  x, shape = (batch_size, 2, embedding_dim)
    输出: logits, shape = (batch_size, num_classes)
    """
    def __init__(
        self,
        embedding_dim: int,
        num_classes: int = 3,
        hidden_dim: int = 128,      # 隐层大小，可自行调整
        num_hidden_layers: int = 2  # 隐层层数，可自行调整
    ):
        super().__init__()

        input_dim = 2 * embedding_dim  # 先把两个 embedding 展平拼接

        layers = []
        # 第一层：input_dim -> hidden_dim
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        # 中间的隐藏层
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        self.mlp = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        x: (batch_size, 2, embedding_dim)
        """
        # 展平两个 embedding： (B, 2, D) -> (B, 2*D)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        logits = self.classifier(x)
        return logits