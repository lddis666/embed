# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from Dataset import ASCategoryDataset, ASEmbeddingLoader
from Model import ASClassifier
import re


    

class ASClassificationPipeline:
    def __init__(
        self, 
        ds, 
        emb_loader, 
        batch_size=512, 
        val_ratio=0.1, 
        test_ratio=0.1, 
        embedding_dim=16, 
        lr=5e-3,
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
                label_batch = label_batch.to(self.device)
                logits = self.model(asn_batch)
                preds = torch.argmax(logits, dim=1)
                y_true.extend(label_batch.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
        acc = accuracy_score(y_true, y_pred)
        f1_str = classification_report(y_true, y_pred,  zero_division=0)
        report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        return acc, f1_str, report_dict

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
    






# ### NEW CODE ###

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# -----------------------------
# 1. Embedding 列表
# -----------------------------
embedding_files = [
    "./dataset/bgp2vec-embeddings.txt",
    "./dataset/node2vec-embeddings16-10-100.txt",
    "./output/as_contextual_embedding.txt",
    "./output/as_static_embedding.txt", 
    "./dataset/beam.txt",
    "./output/as_contextual_embedding_only_map.txt",
    "./bgp2vec/bgp2vec_asn_embeddings.txt"
]

# -----------------------------
# 2. 分类任务列表
# -----------------------------
categories = [
    'continent',
    'traffic_ratio',
    'scope',
    'network_type',
    'policy',
    'industry'
]

# ===========================================================
#          函数：从 classification_report 中提取 macro F1
# ===========================================================


    
def extract_macro_f1(report: str):
    match = re.search(r"macro avg\s+\S+\s+\S+\s+(\S+)", report)
    return float(match.group(1)) if match else None

def extract_weighted_f1(report: str):
    match = re.search(r"weighted avg\s+\S+\s+\S+\s+(\S+)", report)
    return float(match.group(1)) if match else None

def extract_accuracy(report: str):
    # accuracy 行通常为：accuracy  <spaces> 0.85
    match = re.search(r"accuracy\s+(\S+)", report)
    return float(match.group(1)) if match else None


# ===========================================================
#      读取 embedding 文件，获取 ASN 集合
# ===========================================================

def load_as_set(path):
    aset = set()
    with open(path, 'r') as f:
        header = f.readline()
        for line in f:
            asn = line.strip().split(',')[0]
            if asn.isdigit():
                aset.add(int(asn))
    return aset


# 初始化为 None，用第一个 embedding 集合作为起点
all_as_set = None

for emb_file in embedding_files:
    aset = load_as_set(emb_file)
    print(f"Loaded {len(aset)} ASNs from {emb_file}")

    if all_as_set is None:
        all_as_set = aset        # 第一个集合
    else:
        all_as_set &= aset       # 逐步求交集

print(f"\nTotal ASN union size = {len(all_as_set)}\n")


# ===========================================================
#           主循环：遍历 embedding × category
# ===========================================================

results = {}   # 保存所有结果

for emb_path in embedding_files:

    print("\n==============================")
    print(f"  Testing Embedding: {emb_path}")
    print("==============================")

    emb_loader = ASEmbeddingLoader(emb_path, device=device)
    embedding_dim = len(list(emb_loader.asn_to_embedding.values())[0])

    results[emb_path] = {}

    for cat in categories:

        print(f"\n--- Category: {cat} ---")

        ds = ASCategoryDataset(
            './node_features.csv',
            category=cat,
            min_count=500,
            to_merge=True,
            embedding_loader=emb_loader,
            filter_asns=all_as_set
        )

        pipeline = ASClassificationPipeline(
            ds,
            emb_loader,
            batch_size=512,
            val_ratio=0.1,
            test_ratio=0.1,
            embedding_dim=embedding_dim
        )

        pipeline.train(epochs=50)

        _, report, report_dict = pipeline.evaluate(split='test')


        
        macro_f1 = report_dict['macro avg']['f1-score']
        weighted_f1 = report_dict['weighted avg']['f1-score']
        acc = report_dict['accuracy']

        print("\nClassification Report:")
        print(report)
        print(f"Extracted Accuracy      = {acc:.4f}")
        print(f"Extracted Macro F1      = {macro_f1:.4f}")
        print(f"Extracted Weighted F1   = {weighted_f1:.4f}")

        results[emb_path][cat] = {
            "acc": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "report": report
        }

# ===========================================================
#             输出最终 Summary（acc, macro, weighted）
# ===========================================================

print("\n\n===== Summary Results =====\n")

for emb_path, cat_results in results.items():
    print(f"\nEmbedding: {emb_path}")
    for cat, data in cat_results.items():
        print(f"  {cat:15s}: "
              f"acc={data['acc']:.4f}, "
              f"macro_f1={data['macro_f1']:.4f}, "
              f"weighted_f1={data['weighted_f1']:.4f}")
        




# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# def extract_macro_f1(report_str):
#     """
#     从 sklearn classification_report 的字符串中提取 macro avg F1-score
#     """
#     for line in report_str.splitlines():
#         if "macro avg" in line:
#             parts = re.split(r"\s+", line.strip())
#             # macro avg 行一般格式为:
#             # macro avg    precision    recall    f1-score    support
#             f1 = float(parts[-2])
#             return f1
#     return None



# # emb_loader = ASEmbeddingLoader("./dataset/bgp2vec-embeddings.txt", device=device)
# emb_loader = ASEmbeddingLoader("./dataset/node2vec-embeddings16-10-100.txt", device=device)
# emb_loader = ASEmbeddingLoader("./output/as_contextual_embedding.txt", device=device)
# # emb_loader = ASEmbeddingLoader("./output/as_static_embedding.txt", device=device)



# ds =  ASCategoryDataset('./node_features.csv', category='continent',      min_count=1000, to_merge=True, embedding_loader=emb_loader)
# ds =  ASCategoryDataset('./node_features.csv', category='traffic_ratio',  min_count=1000, to_merge=True, embedding_loader=emb_loader)
# ds =  ASCategoryDataset('./node_features.csv', category='scope',          min_count=1000, to_merge=True, embedding_loader=emb_loader)
# ds =  ASCategoryDataset('./node_features.csv', category='network_type',   min_count=1000, to_merge=True, embedding_loader=emb_loader)
# # ds =  ASCategoryDataset('./node_features.csv', category='policy',         min_count=1000, to_merge=True, embedding_loader=emb_loader)
# # ds =  ASCategoryDataset('./node_features.csv', category='industry',       min_count=1000, to_merge=True, embedding_loader=emb_loader)



# pipeline = ASClassificationPipeline(
#     ds, 
#     emb_loader, 
#     batch_size=32, 
#     val_ratio=0.1, 
#     test_ratio=0.1, 
#     embedding_dim=len(list(emb_loader.asn_to_embedding.values())[0])
# )

# pipeline.train(epochs=3)
# print('Test F1:\n', pipeline.evaluate(split='test')[1])

