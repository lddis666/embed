# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from sklearn.metrics import accuracy_score
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from Dataset import ASCategoryDataset, ASEmbeddingLoader, ASRelationDataset, LinkPredictionDataset
from Model import ASClassifier, ToRClassifier
import re
import random
from collections import defaultdict

def split_train_valid_test_indices(
    labels,
    train_ratio=0.7,
    valid_ratio=0.15,
    test_ratio=0.15,
    seed=42
):
    """
    按类别分层划分 train/valid/test，且互不重叠。
    
    参数:
      labels: 长度为 N 的一维列表/数组，对应 dataset.labels
      train_ratio, valid_ratio, test_ratio: 三个比例之和应约等于 1
      seed: 随机种子
    
    返回:
      train_indices, valid_indices, test_indices: 三个索引列表
    """
    assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6

    random.seed(seed)
    class_to_indices = defaultdict(list)
    for idx, y in enumerate(labels):
        class_to_indices[y].append(idx)

    train_indices, valid_indices, test_indices = [], [], []

    for c, idxs in class_to_indices.items():
        idxs = idxs[:]           # 复制一份
        random.shuffle(idxs)     # 类内打乱
        
        n = len(idxs)
        n_train = int(round(n * train_ratio))
        n_valid = int(round(n * valid_ratio))
        # 剩下全部给 test
        n_test = n - n_train - n_valid
        
        c_train = idxs[:n_train]
        c_valid = idxs[n_train:n_train + n_valid]
        c_test  = idxs[n_train + n_valid:]
        
        train_indices.extend(c_train)
        valid_indices.extend(c_valid)
        test_indices.extend(c_test)

    # 整体再打乱一下
    random.shuffle(train_indices)
    random.shuffle(valid_indices)
    random.shuffle(test_indices)

    return train_indices, valid_indices, test_indices

def make_balanced_subset_from_indices(dataset, base_indices, frac=0.8, min_per_class=1, seed=42):
    """
    在给定的 base_indices（例如 train_indices）上做平衡下采样。
    
    参数:
      dataset: ASCategoryDataset 实例
      base_indices: 要在其内部进行下采样的一组索引（如 train_indices）
      frac: 以该集合内最小类别样本数的 frac 倍作为每类目标数
      min_per_class: 每类最少保留样本数
      seed: 随机种子
    
    返回:
      balanced_indices: 新的、平衡后的索引（都在 base_indices 里）
      per_class_n: dict，记录每个类别采样多少
    """
    random.seed(seed)
    labels = dataset.labels

    # 1. 只看 base_indices 中的样本，按类收集索引
    class_to_indices = defaultdict(list)
    for idx in base_indices:
        y = labels[idx]
        class_to_indices[y].append(idx)

    # 2. 计算该子集内最小类别大小
    class_sizes = {c: len(idxs) for c, idxs in class_to_indices.items()}
    min_size = min(class_sizes.values())

    target_per_class = max(int(round(min_size * frac)), min_per_class)

    balanced_indices = []
    per_class_n = {}
    for c, idxs in class_to_indices.items():
        if len(idxs) <= target_per_class:
            chosen = idxs
        else:
            chosen = random.sample(idxs, target_per_class)
        balanced_indices.extend(chosen)
        per_class_n[c] = len(chosen)

    random.shuffle(balanced_indices)
    return balanced_indices, per_class_n


    
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # Tensor: [num_classes], 类似 class weight
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(weight=alpha, reduction='none')

    def forward(self, logits, target):
        ce_loss = self.ce(logits, target)  # [B]
        pt = torch.exp(-ce_loss)           # 预测正确的概率
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class ASClassificationPipeline:
    def __init__(
        self, 
        ds, 
        emb_loader, 
        batch_size=512, 
        val_ratio=0.1, 
        test_ratio=0.1, 
        embedding_dim=16, 
        lr=5e-5,
        device=None,
        seed=42,
        single_type = True, 
    ):
        self.ds = ds
        self.emb_loader = emb_loader
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.lr = lr
        self.seed = seed
        self.single_type = single_type
        torch.manual_seed(seed)
        numpy_rng = np.random.default_rng(seed)
        # dataset split
        # n_total = len(ds)
        # n_test = int(n_total * test_ratio)
        # n_val = int(n_total * val_ratio)
        # n_train = n_total - n_test - n_val
        # self.train_ds, self.val_ds, self.test_ds = random_split(ds, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(seed))


        labels = ds.labels   # 来自你的 ASCategoryDataset

        train_indices, valid_indices, test_indices = split_train_valid_test_indices(
            labels,
            train_ratio = 1 - test_ratio - val_ratio,
            valid_ratio = val_ratio,
            test_ratio = test_ratio,
            seed=42
        )

        # print(len(train_indices), len(valid_indices), len(test_indices))

        # balanced_train_indices, per_class_n = make_balanced_subset_from_indices(
        #     ds,
        #     base_indices=train_indices,
        #     frac=1,        # 使用最小类别样本数的 0.8
        #     min_per_class=1,
        #     seed=42
        # )

        # print("平衡后的各类样本数：", per_class_n)
        # print("平衡后的训练集大小：", len(balanced_train_indices))

        from torch.utils.data import Subset, DataLoader

        self.train_ds = Subset(ds, train_indices)  
        self.val_ds = Subset(ds, valid_indices)           
        self.test_ds  = Subset(ds, test_indices)            


        # # 先拿到原始数据集中每个 index 的 label（注意 random_split 会打乱索引）
        # train_labels = []
        # for idx in self.train_ds.indices:  # Subset.indices
        #     _, label = self.ds[idx]
        #     train_labels.append(label)
        # train_labels = np.array(train_labels)

        # # 用和 class_weights 一致的逻辑构造 sample_weights
        # label_counts = np.bincount(train_labels, minlength=len(ds.get_label_map()))
        # freq = label_counts / label_counts.sum()
        # inv_freq = 1.0 / (freq + 1e-6)
        # weights_per_class = np.sqrt(inv_freq)
        # weights_per_class = weights_per_class / weights_per_class.mean()

        # sample_weights = weights_per_class[train_labels]  # 每个样本的权重
        # sample_weights = torch.tensor(sample_weights, dtype=torch.float)

        # sampler = WeightedRandomSampler(
        #     weights=sample_weights,
        #     num_samples=len(sample_weights),  # 每个 epoch 采样与训练集大小相同
        #     replacement=True
        # )

        # self.train_loader = DataLoader(
        #     self.train_ds,
        #     batch_size=batch_size,
        #     sampler=sampler,   # 注意：有 sampler 时就不要 shuffle=True
        #     drop_last=False
        # )
        self.train_loader = DataLoader(self.train_ds, batch_size=batch_size, shuffle=True)



        self.val_loader = DataLoader(self.val_ds, batch_size=batch_size)
        self.test_loader = DataLoader(self.test_ds, batch_size=batch_size)

        self.best_val_acc = -1.0
        self.best_weighted_f1 = -1.0
        self.best_state_dict = None


        # 统计类别样本数
        label_counts = np.zeros(len(ds.get_label_map()))
        for _, label in self.ds:
            label_counts[label] += 1
        # for循环分别打印每个类别的数量
        for i, count in enumerate(label_counts):
            print(f"类别 {i} 样本数: {count}")



        # # 计算加权交叉熵的权重，常用方式：inverse frequency
        # weights = 1.0 / (label_counts + 1e-6)
        # weights = weights / weights.sum() * len(weights)  # 归一化使平均权重为1

        # freq = label_counts / label_counts.sum()
        # inv_freq = 1.0 / (freq + 1e-6)
        # weights = np.log1p(1 + inv_freq) # 或 np.log(1 + inv_freq)
        # weights = weights / weights.mean()

        # class_weights = torch.tensor(weights, dtype=torch.float, device=self.device)


        freq = label_counts / label_counts.sum()
        weights = 1.0 / (freq + 1e-6)

        # 限制一个最大/最小，避免极端
        max_w = 10.0
        min_w = 0.1
        weights = np.clip(weights, min_w, max_w)

        # 可以不归一化，也可以简单归一化到均值 1
        weights = weights / weights.mean()

        class_weights = torch.tensor(weights, dtype=torch.float, device=self.device)

        # 网络结构
        num_classes = len(ds.get_label_map())
        if self.single_type:
            self.model = ASClassifier(embedding_dim, num_classes).to(self.device)
        else:
            self.model = ToRClassifier(embedding_dim, num_classes).to(self.device)
        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        # self.criterion = FocalLoss(alpha=class_weights, gamma=2.0, reduction='mean')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.label_map = ds.get_label_map()  # 标签编号到类别名映射

    def _batch_asn_to_embedding(self, asn_batch):
        # 输入LongTensor 转成 list，然后查embeddings
        asn_list = asn_batch.cpu().tolist()
        emb_tensor = self.emb_loader.get_batch(asn_list).to(self.device)  # shape (B, D)
        return emb_tensor
    
    def load_best_model(self):
        """将训练过程中在验证集上表现最好的模型权重加载回 model。"""
        if self.best_state_dict is None:
            raise ValueError("Best model state_dict is None. 请先调用 train，并确保有至少一个 epoch 的验证结果。")
        self.model.load_state_dict(self.best_state_dict)
        self.model.to(self.device)
        

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


                # 如果想用 macro F1 做标准，可以这样得到：
                # val_macro_f1 = val_report_dict['macro avg']['f1-score']


                    # 如果希望同步存盘，可以再写一行：
                    # torch.save(self.best_state_dict, 'best_model.pt')

            if epoch % print_interval == 0:

                # 每个 epoch 之后在验证集上评估一次
                val_acc, _, val_report_dict = self.evaluate(split='val')
                print(f'Epoch {epoch}, Loss={np.mean(losses):.4f}, Val accuracy={val_acc:.4f}, Best Val acc={self.best_val_acc:.4f}')

                # 以 val_acc 作为最佳模型标准
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    # 只保存权重即可
                    self.best_state_dict = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}


            # if epoch % print_interval == 0:
            #     val_acc = self.evaluate(split='val')[0]
            #     print(f'Epoch {epoch}, Loss={np.mean(losses):.4f}, Val accuracy={val_acc:.4f}')

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

    # def predict(self, asn_list):
    #     # asn_list: list of ASN (int)
    #     self.model.eval()
    #     embs = self.emb_loader.get_batch(asn_list).to(self.device)
    #     with torch.no_grad():
    #         logits = self.model(embs)
    #         preds = torch.argmax(logits, dim=1)
    #     # 返回类别编号和类别名
    #     idx2label = self.label_map

    #     # print(idx2label)

    #     class_names = [idx2label[p.item()] for p in preds]
    #     return preds.cpu().tolist(), class_names

    def get_label_map(self):
        return self.label_map
    






# ### NEW CODE ###

device = 'cuda' if torch.cuda.is_available() else 'cpu'



# -----------------------------
# 1. Embedding 列表
# -----------------------------
embedding_files = [
    # "./feature_embeddings_cls.csv",

    # ----
    "./deepwalk/deepwalk_embeddings.txt",
    "./deepwalk/node2vec_embeddings.txt",
    "./bgp2vec/bgp2vec_asn_embeddings.txt",
    "./dataset/beam.txt",
    # ----

    # "./output/as_contextual_embedding_1201-map-mrf-with-feat.txt",
    # "./output/as_contextual_embedding_1201-map-mfr-without-feat.txt",
    # "./dataset/bgp2vec-embeddings.txt",
    # "./dataset/node2vec-embeddings16-10-100.txt",
    # "./output/as_contextual_embedding.txt",
    # "./output/as_static_embedding.txt", 
    
    # "./output/as_contextual_embedding_only_map.txt",

    # "./output/as_static_embedding_only_map.txt",
    # "./output/as_static_embedding_1127.txt", # 只有任务， 20 epoch, lr 1e-4
    # "./output/as_contextual_embedding_1127.txt",
    # "./output/as_contextual_embedding_1128-map-mfr-without-feat.txt",
    # "./output/as_contextual_embedding_1128-only-map-without-feat.txt",
    # "./output/as_contextual_embedding_1128-map-mfr-with-feat.txt"



    # "./output/as_contextual_embedding_1129-map-mfr-with-feat.txt",
    # "./output/as_contextual_embedding_1129-map-mfr-without-feat.txt",
    # "./output/as_contextual_embedding_1129-map-with-feat.txt",
    # "./output/as_contextual_embedding_1129-map-without-feat.txt",
    # "./output/as_contextual_embedding_1130-map-mfr-without-feat.txt",
    # "./output/as_contextual_embedding_1130-map-mrf-with-feat.txt",

    # "./output/as_contextual_embedding_1202-map-mfr-without-feat-1.txt",
    # "./output/as_contextual_embedding_1202-map-mrf-with-feat-1.txt",

    # "./output/as_static_embedding_1202-map-mfr-without-feat-1.txt",
    # "./output/as_static_embedding_1202-map-mrf-with-feat-1.txt",


    # "./output/as_contextual_embedding_1202-map-mfr-without-feat-2.txt",
    # "./output/as_contextual_embedding_1202-map-mrf-with-feat-2.txt",

    # "./output/as_static_embedding_1202-map-mfr-without-feat-2.txt",
    # "./output/as_static_embedding_1202-map-mrf-with-feat-2.txt",

    # "./output/as_contextual_embedding_1202-map-mfr-without-feat-3.txt",
    # "./output/as_contextual_embedding_1202-map-mrf-with-feat-3.txt",

    # "./output/as_static_embedding_1202-map-mfr-without-feat-3.txt",
    # "./output/as_static_embedding_1202-map-mrf-with-feat-3.txt",





    # "./output/as_contextual_embedding_1202-map-without-feat-3.txt",
    # "./output/as_contextual_embedding_1202-map-with-feat-3.txt",
    # "./output/as_static_embedding_1202-map-without-feat-3.txt",
    # "./output/as_static_embedding_1202-map-with-feat-3.txt",


    # "./output/as_contextual_embedding_1203-map-mfr-without-feat-3.txt",
    # "./output/as_contextual_embedding_1203-map-mrf-with-feat-3.txt",

    # "./output/as_static_embedding_1203-map-mfr-without-feat-3.txt",
    # "./output/as_static_embedding_1203-map-mrf-with-feat-3.txt",

    # "./output/as_contextual_embedding_1203-map-mrf-with-feat-4.txt",
    # "./output/as_contextual_embedding_1203-map-mfr-without-feat-4.txt",

    # "./output/as_static_embedding_1203-map-mrf-with-feat-4.txt",
    # "./output/as_static_embedding_1203-map-mfr-without-feat-4.txt",

    # "./output/as_static_embedding_1203-map-mfr-no-missing-indicator.txt",
    # "./output/as_contextual_embedding_1203-map-mfr-no-missing-indicator.txt",

    # "./output/as_contextual_embedding_1203-map-mfr-no-missing-indicator-200_lambda-50_epoch.txt",
    # "./output/as_static_embedding_1203-map-mfr-no-missing-indicator-200_lambda-50_epoch.txt",
    "./output/as_contextual_embedding_1203-map-mfr-no-missing-indicator-4_lambda-40_epoch.txt",
    "./output/as_static_embedding_1203-map-mfr-no-missing-indicator-4_lambda-40_epoch.txt",
    "./output/as_contextual_embedding_1206-no-map.txt",
    "./output/as_contextual_embedding_1206-no-mfr-new.txt",
    # "./output/as_contextual_embedding_1203-merge.txt"



    
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
    'industry',
    'as_relation',
    'link_prediction',
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

        if cat == 'as_relation' :
            ds = ASRelationDataset(
                csv_path='./dataset/as_relations_onehot.txt',
                relation_fields=("P2P", "P2C", "C2P"),
                # relation_fields=("P2P", "P2C"),
                embedding_loader=emb_loader,  
                filter_asns=all_as_set,
            )
            bs = 2048
            lr = 1e-4

        elif cat =='link_prediction':
            ds = LinkPredictionDataset(
                csv_path='./dataset/raw_edges.csv',
                embedding_loader=emb_loader,  
                filter_nodes=all_as_set,
                num_pos_samples = 10000,
                negative_ratio = 2.0,
                undirected=False,
            )
            bs = 2048
            lr = 1e-4
        else:
            min_count = 800
            ds = ASCategoryDataset(
                './node_features.csv',
                category=cat,
                min_count=  min_count,
                to_merge=True,
                embedding_loader=emb_loader,
                filter_asns=all_as_set
            )
            bs = 256
            lr = 5e-5

            print(f"--- Dataset: {cat} ---")
            print(f"  Total ASNs: {len(ds)}")
            print(f"  Label Map: {ds.get_label_map()}")

            print("final_fields:", ds.final_fields)
            print("label_map:", ds.get_label_map())
            print("label_count:", ds.get_label_count())


            # print("final_fields:", ds.final_fields)
            # print("label_map:", ds.get_label_map())
            # print("label_count:", ds.get_label_count())
            # print("min label_count:", min(ds.get_label_count().values()))

        pipeline = ASClassificationPipeline(
            ds,
            emb_loader,
            batch_size=bs,
            val_ratio=0.1,
            test_ratio=0.1,
            lr = lr,
            embedding_dim=embedding_dim,
            single_type=False if cat in ['as_relation', 'link_prediction'] else True
        )

        pipeline.train(epochs=20, print_interval=1)

        # 加载验证集表现最好的模型
        pipeline.load_best_model()

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

# 保存reslust
import pickle
with open("results.pkl", "wb") as f:
    pickle.dump(results, f)
        
import pandas as pd

with open("results.pkl", "rb") as f:
    new_data = pickle.load(f)


# ------------------------------------------------------
# 1) 解析新的嵌套数据结构 → 转成 rows 列表
# ------------------------------------------------------
rows = []

for embedding, tasks_dict in new_data.items():  # 每个 embedding 对应一个文件
    
    emb_name = embedding.split("/")[-1].replace(".csv", "").replace(".txt", "")
    
    for task, metrics_dict in tasks_dict.items():  # 每个任务（continent, scope, ...）
        for metric, val in metrics_dict.items():
            if metric == "report":  
                continue  # 不需要报告内容
            
            rows.append({
                "embedding": emb_name,
                "metric": metric,
                "task": task,
                "score": val
            })

# ------------------------------------------------------
# 2) 生成新的 DataFrame（与旧格式完全兼容）
# ------------------------------------------------------
long_df = pd.DataFrame(rows)
long_df = long_df[long_df["metric"] == "weighted_f1"]
long_df_temp = long_df.copy()

# ------------------------------------------------------
# 3) 计算 ranking（按 metric + task 分组）
# ------------------------------------------------------
long_df["rank"] = long_df.groupby(["metric", "task"])["score"] \
                         .rank(ascending=False, method="average")


# ------------------------------------------------------
# 4) 每个 embedding 的平均排名
# ------------------------------------------------------
avg_rank = long_df.groupby("embedding")["rank"].mean().sort_values()

print("==== 每个 embedding 的平均排名（越小越好） ====")
print(avg_rank, "\n")



# ====================================================
# 1. 在每个 (metric, task) 组内做 Min-Max 归一化
#    归一化后列名：norm_score
# ====================================================
def min_max_group(g):
    s_min = g["score"].min()
    s_max = g["score"].max()
    if s_max == s_min:
        # 所有 embedding 在这个 metric+task 上得分一样，
        # 归一化后全设为 1（或者 0.5 也行，看你习惯）
        g["norm_score"] = 1.0
    else:
        g["norm_score"] = (g["score"] - s_min) / (s_max - s_min)
    return g


long_df = long_df_temp

long_df = long_df.groupby(["metric", "task"], group_keys=False).apply(min_max_group)




# ====================================================
# 2. 按 embedding 聚合：
#    - AvgScore_i   : norm_score 的平均值
#    - MinScore_i   : norm_score 的最小值（最差任务）
#    - Std_i        : norm_score 的标准差（衡量不均衡）
#    - BalancedScore_i = AvgScore_i - lambda * Std_i
# ====================================================
lambda_ = 0.5  # 可根据需要调整

agg_df = (
    long_df
    .groupby("embedding")["norm_score"]
    .agg(AvgScore="mean",
         MinScore="min",
         Std="std")
    .reset_index()
)

# 可能某些 embedding 在所有 task 上分数一样 → Std 为 NaN，设为 0
agg_df["Std"] = agg_df["Std"].fillna(0.0)

agg_df["BalancedScore"] = agg_df["AvgScore"] - lambda_ * agg_df["Std"]

# 按 BalancedScore 从高到低排序（也可以按 AvgScore 排）
agg_df = agg_df.sort_values(by="BalancedScore", ascending=False)



# ====================================================
# 3. 如果你想单独看某个指标的排名（可选）
#    例如按 AvgScore 排序：
# ====================================================
print("==== 按 AvgScore 排序（越大越好） ====")
print(agg_df.sort_values("AvgScore", ascending=False)[
    ["embedding", "AvgScore", "MinScore", "Std", "BalancedScore"]
])









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

