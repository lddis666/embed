import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from Dataset import ASCategoryDataset, ASEmbeddingLoader
from Model import ASClassifier


    

class ASClassificationPipeline:
    def __init__(
        self, 
        ds, 
        emb_loader, 
        batch_size=64, 
        val_ratio=0.1, 
        test_ratio=0.1, 
        embedding_dim=16, 
        lr=1e-3,
        device=None,
        seed=42,
    ):
        self.ds = ds
        self.emb_loader = emb_loader
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.lr = lr
        self.seed = seed
        torch.manual_seed(seed)
        numpy_rng = np.random.default_rng(seed)
        # dataset split
        n_total = len(ds)
        n_test = int(n_total * test_ratio)
        n_val = int(n_total * val_ratio)
        n_train = n_total - n_test - n_val
        self.train_ds, self.val_ds, self.test_ds = random_split(ds, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(seed))
        self.train_loader = DataLoader(self.train_ds, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_ds, batch_size=batch_size)
        self.test_loader = DataLoader(self.test_ds, batch_size=batch_size)


        # 统计类别样本数
        label_counts = np.zeros(len(ds.get_label_map()))
        for _, label in self.ds:
            label_counts[label] += 1

        # 计算加权交叉熵的权重，常用方式：inverse frequency
        weights = 1.0 / (label_counts + 1e-6)
        weights = weights / weights.sum() * len(weights)  # 归一化使平均权重为1
        class_weights = torch.tensor(weights, dtype=torch.float, device=self.device)


        # 网络结构
        num_classes = len(ds.get_label_map())
        self.model = ASClassifier(embedding_dim, num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.label_map = ds.get_label_map()  # 标签编号到类别名映射

    def _batch_asn_to_embedding(self, asn_batch):
        # 输入LongTensor 转成 list，然后查embeddings
        asn_list = asn_batch.cpu().tolist()
        emb_tensor = self.emb_loader.get_batch(asn_list).to(self.device)  # shape (B, D)
        return emb_tensor

    def train(self, epochs=10, print_interval=1):
        print(len(self.train_loader))
        for epoch in range(1, epochs+1):
            self.model.train()
            losses = []
            for asn_batch, label_batch in self.train_loader:
                # 获取embedding
                label_batch = label_batch.to(self.device)
                logits = self.model(asn_batch)
                loss = self.criterion(logits, label_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
            if epoch % print_interval == 0:
                val_acc = self.evaluate(split='val')[0]
                print(f'Epoch {epoch}, Loss={np.mean(losses):.4f}, Val accuracy={val_acc:.4f}')

    def evaluate(self, split='test'):
        self.model.eval()
        loader = {
            'val': self.val_loader,
            'test': self.test_loader
        }[split]
        y_true = []
        y_pred = []
        with torch.no_grad():
            for asn_batch, label_batch in loader:
                # emb = self._batch_asn_to_embedding(asn_batch)
                label_batch = label_batch.to(self.device)
                logits = self.model(asn_batch)
                preds = torch.argmax(logits, dim=1)
                y_true.extend(label_batch.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
        acc = accuracy_score(y_true, y_pred)
        f1_str = classification_report(y_true, y_pred,  zero_division=0)
        return acc, f1_str

    def predict(self, asn_list):
        # asn_list: list of ASN (int)
        self.model.eval()
        embs = self.emb_loader.get_batch(asn_list).to(self.device)
        with torch.no_grad():
            logits = self.model(embs)
            preds = torch.argmax(logits, dim=1)
        # 返回类别编号和类别名
        idx2label = self.label_map

        # print(idx2label)

        class_names = [idx2label[p.item()] for p in preds]
        return preds.cpu().tolist(), class_names

    def get_label_map(self):
        return self.label_map
    




# emb_loader = ASEmbeddingLoader("/Users/ldd/Desktop/embed/dataset/bgp2vec-embeddings.txt", device="cpu")
emb_loader = ASEmbeddingLoader("/Users/ldd/Desktop/embed/dataset/node2vec-embeddings16-10-100.txt", device="cpu")
# emb_loader = ASEmbeddingLoader("/Users/ldd/Desktop/embed/dataset/deepwalk-embeddings-wl100-ws-10.txt", device="cpu")

# ds = ASCategoryDataset('/Users/ldd/Desktop/embed/node_features.csv', category='industry', min_count=5000, to_merge=True,embedding_loader =  emb_loader)

# ds =  ASCategoryDataset('/Users/ldd/Desktop/embed/node_features.csv', category='continent',      min_count=500, to_merge=True, embedding_loader=emb_loader)
# ds =  ASCategoryDataset('/Users/ldd/Desktop/embed/node_features.csv', category='traffic_ratio',  min_count=500, to_merge=True, embedding_loader=emb_loader)
# ds =  ASCategoryDataset('/Users/ldd/Desktop/embed/node_features.csv', category='scope',          min_count=500, to_merge=True, embedding_loader=emb_loader)
# ds =  ASCategoryDataset('/Users/ldd/Desktop/embed/node_features.csv', category='network_type',   min_count=500, to_merge=True, embedding_loader=emb_loader)
ds =  ASCategoryDataset('/Users/ldd/Desktop/embed/node_features.csv', category='policy',         min_count=500, to_merge=True, embedding_loader=emb_loader)
# ds =  ASCategoryDataset('/Users/ldd/Desktop/embed/node_features.csv', category='industry',       min_count=500, to_merge=True, embedding_loader=emb_loader)


print(len(ds))

pipeline = ASClassificationPipeline(
    ds, 
    emb_loader, 
    batch_size=32, 
    val_ratio=0.1, 
    test_ratio=0.1, 
    embedding_dim=16
)

pipeline.train(epochs=20)
print('Test F1:\n', pipeline.evaluate(split='test')[1])

# 预测
asn_list = [3356, 6939, 1299]
preds, class_names = pipeline.predict(asn_list)
print('预测类别ID:', preds)
print('预测类别名:', class_names)
print('类别映射:', pipeline.get_label_map())


