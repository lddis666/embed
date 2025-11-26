import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

# 各类别对应字段
CATEGORIES = {
    "continent": [
        "AS_rank_continent_Africa",
        "AS_rank_continent_Asia",
        "AS_rank_continent_Europe",
        # "AS_rank_continent_None",
        "AS_rank_continent_North America",
        "AS_rank_continent_Oceania",
        "AS_rank_continent_South America",
    ],
    "traffic_ratio": [
        "peeringDB_info_ratio_Balanced",
        "peeringDB_info_ratio_Heavy Inbound",
        "peeringDB_info_ratio_Heavy Outbound",
        "peeringDB_info_ratio_Mostly Inbound",
        "peeringDB_info_ratio_Mostly Outbound",
        # "peeringDB_info_ratio_None",
        "peeringDB_info_ratio_Not Disclosed",
    ],
    "scope": [
        "peeringDB_info_scope_Africa",
        "peeringDB_info_scope_Asia Pacific",
        "peeringDB_info_scope_Australia",
        "peeringDB_info_scope_Europe",
        "peeringDB_info_scope_Global",
        "peeringDB_info_scope_Middle East",
        # "peeringDB_info_scope_None",
        "peeringDB_info_scope_North America",
        "peeringDB_info_scope_Not Disclosed",
        "peeringDB_info_scope_Regional",
        "peeringDB_info_scope_South America",
    ],
    "network_type": [
        "peeringDB_info_type_Cable/DSL/ISP",
        "peeringDB_info_type_Content",
        "peeringDB_info_type_Educational/Research",
        "peeringDB_info_type_Enterprise",
        "peeringDB_info_type_Government",
        "peeringDB_info_type_NSP",
        "peeringDB_info_type_Network Services",
        "peeringDB_info_type_Non-Profit",
        # "peeringDB_info_type_None",
        "peeringDB_info_type_Not Disclosed",
        "peeringDB_info_type_Route Collector",
        "peeringDB_info_type_Route Server",
    ],
    "policy": [
        "peeringDB_policy_general_No",
        # "peeringDB_policy_general_None",
        "peeringDB_policy_general_Open",
        "peeringDB_policy_general_Restrictive",
        "peeringDB_policy_general_Selective",
    ],
    "industry": [
        "ASDB_C1L1_Agriculture, Mining, and Refineries (Farming, Greenhouses, Mining, Forestry, and Animal Farming)",
        "ASDB_C1L1_Community Groups and Nonprofits",
        "ASDB_C1L1_Computer and Information Technology",
        "ASDB_C1L1_Construction and Real Estate",
        "ASDB_C1L1_Education and Research",
        "ASDB_C1L1_Finance and Insurance",
        "ASDB_C1L1_Freight, Shipment, and Postal Services",
        "ASDB_C1L1_Government and Public Administration",
        "ASDB_C1L1_Health Care Services",
        "ASDB_C1L1_Manufacturing",
        "ASDB_C1L1_Media, Publishing, and Broadcasting",
        "ASDB_C1L1_Museums, Libraries, and Entertainment",
        # "ASDB_C1L1_None",
        "ASDB_C1L1_Other",
        "ASDB_C1L1_Retail Stores, Wholesale, and E-commerce Sites",
        "ASDB_C1L1_Service",
        "ASDB_C1L1_Travel and Accommodation",
        "ASDB_C1L1_Utilities (Excluding Internet Service)",
    ]
}

class ASCategoryDataset(Dataset):
    def __init__(self, csv_path, category, min_count=1, to_merge=False, embedding_loader=None, filter_asns=None):
        """
        csv_path: csv文件路径
        category: 选择哪一种分类类别（对应CATEGORIES的key）
        min_count: 类别不足min_count的合并归为一类
        to_merge: 是否合并数量太少的类别
        embedding_loader: 提供 ASN → embedding 的 loader
        filter_asns: 可选, 传入一个 ASN 列表/集合，仅保留这些 ASN
        """
        self.df = pd.read_csv(csv_path)

        print(len(self.df))
        self.category = category
        self.fields = CATEGORIES[category]

        self.embedding_loader = embedding_loader

        self.ASNs = self.df['ASN'].astype(int).tolist()

        # 如果有embedding_loader，过滤没有embedding的AS
        if self.embedding_loader is not None:
            has_embedding = [asn for asn in self.ASNs if asn in self.embedding_loader]
            # 保留这些ASN对应的行
            self.df = self.df[self.df['ASN'].astype(int).isin(has_embedding)].reset_index(drop=True)
            print(len(self.df))
            self.ASNs = self.df['ASN'].astype(int).tolist()

        # 再过滤用户传进来的 filter_asns
        if filter_asns is not None:
            filter_asns = set(filter_asns)  # 支持 list/tuple 等
            before = len(self.df)
            self.df = self.df[self.df['ASN'].astype(int).isin(filter_asns)].reset_index(drop=True)
            print(len(filter_asns))
            print(f"Filtered by user list: {before} → {len(self.df)}")
            self.ASNs = self.df['ASN'].astype(int).tolist()

        labels_matrix = self.df[self.fields].values.astype(float)

        # 删除没有类别的AS
        non_zero_rows = np.where(labels_matrix.sum(axis=1) != 0)[0]
        if len(non_zero_rows) < len(self.df):
            print(f"Warning: {len(self.df)-len(non_zero_rows)} rows dropped due to all-zero labels.")
        self.df = self.df.iloc[non_zero_rows].reset_index(drop=True)
        self.ASNs = [self.ASNs[i] for i in non_zero_rows]
        labels_matrix = labels_matrix[non_zero_rows]

        # # ------- 这段是新加的过滤逻辑 -------
        # # 找出所有行的和
        # row_sums = labels_matrix.sum(axis=1)
        # # 找到非全零的行
        # nonzero_mask = row_sums != 0
        # # 只保留非全零的
        # labels_matrix = labels_matrix[nonzero_mask]
        # self.df = self.df[nonzero_mask].reset_index(drop=True)
        # self.ASNs = self.df['ASN'].astype(int).tolist()
        # # -------

        
        # 找出每行对应的类别下标
        label_indices = np.argmax(labels_matrix, axis=1)  # one-hot只有一个为1

        # 合并类别（数量太小的放一起）
        if to_merge and min_count > 1:
            # 统计每一列（类别）的正样本数量
            counts = (labels_matrix == 1).sum(axis=0)
            to_merge_idx = [i for i, c in enumerate(counts) if ("None" not in self.fields[i]) and c < min_count]
            remain_idx = [i for i in range(len(self.fields)) if i not in to_merge_idx]
            merged_label_name = '_'.join([self.fields[i] for i in to_merge_idx]) or 'Merged_Minority_Classes'
            # 新类别名顺序: 保留的+合并的
            self.final_fields = [self.fields[i] for i in remain_idx] + [merged_label_name]
            # 对应老的idx到新的idx的映射
            old_to_new = {}
            for i in range(len(self.fields)):
                if i in remain_idx:
                    old_to_new[i] = remain_idx.index(i)
                else:
                    old_to_new[i] = len(remain_idx)  # 合并到最后一个

            # 新label_indices
            new_label_indices = []
            for idx in label_indices:
                new_label_indices.append(old_to_new[idx])
            self.labels = new_label_indices
        else:
            self.final_fields = self.fields
            self.labels = label_indices.tolist()

    def __len__(self):
        return len(self.ASNs)

    def __getitem__(self, idx):
        asn = int(self.ASNs[idx])
        label = int(self.labels[idx])
    
        if self.embedding_loader is not None:
            embedding = self.embedding_loader.get_embedding(asn)  # (emb_dim,) tensor
            return embedding, label
        else:
            return asn, label

    def get_label_map(self):
        """
        返回: 类别编号到类别名的字典
        """
        return {i: name for i, name in enumerate(self.final_fields)}
    
    def get_label_count(self):
        """
        返回每个类别的AS数量
        """
        from collections import Counter
        return dict(Counter(self.labels))



class ASRelationDataset(Dataset):
    """
    用于 AS 之间关系分类的 Dataset。
    输入：两个 ASN（或它们的 embedding）
    输出：关系类别（如 P2P / P2C / C2P）
    """
    def __init__(
        self,
        csv_path,
        relation_fields=("P2P", "P2C", "C2P"),
        embedding_loader=None,
        filter_asns=None,
        min_count=1,
        to_merge=False,
    ):
        """
        :param csv_path: 关系 CSV 文件路径, 格式类似:
                         ASN1,ASN2,P2P,P2C,C2P
                         1,11537,1,0,0
                         ...
        :param relation_fields: 关系字段名列表，默认 ("P2P","P2C","C2P")
        :param embedding_loader: ASEmbeddingLoader 实例 (可选)
        :param filter_asns: 只保留这些 ASN 相关的关系 (可选, list/set)
        :param min_count: 若 to_merge=True, 小于 min_count 的类别会被合并
        :param to_merge: 是否合并样本数太少的关系类别
        """
        self.df = pd.read_csv(csv_path)
        self.relation_fields = list(relation_fields)
        self.embedding_loader = embedding_loader

        # 确保 ASN1/ASN2 是 int
        self.df["ASN1"] = self.df["ASN1"].astype(int)
        self.df["ASN2"] = self.df["ASN2"].astype(int)

        # 先根据 embedding_loader 过滤：没有 embedding 的 AS 不要
        if self.embedding_loader is not None:
            def has_both_embeddings(row):
                return (int(row["ASN1"]) in self.embedding_loader) and \
                       (int(row["ASN2"]) in self.embedding_loader)
            before = len(self.df)
            self.df = self.df[self.df.apply(has_both_embeddings, axis=1)].reset_index(drop=True)
            print(f"Filtered by embedding_loader: {before} → {len(self.df)} rows")

        # 若指定了 filter_asns，只保留出现的 ASN1/ASN2 在该集合中的样本
        if filter_asns is not None:
            filter_asns = set(int(a) for a in filter_asns)
            before = len(self.df)
            mask = self.df["ASN1"].isin(filter_asns) & self.df["ASN2"].isin(filter_asns)
            self.df = self.df[mask].reset_index(drop=True)
            print(f"Filtered by user ASN list: {before} → {len(self.df)} rows")

        # 取关系标签矩阵
        labels_matrix = self.df[self.relation_fields].values.astype(float)

        # 删除关系全 0 的行（没有任何 P2P/P2C/C2P 标记）
        non_zero_rows = np.where(labels_matrix.sum(axis=1) != 0)[0]
        if len(non_zero_rows) < len(self.df):
            print(f"Warning: {len(self.df) - len(non_zero_rows)} rows dropped due to all-zero relation labels.")
        self.df = self.df.iloc[non_zero_rows].reset_index(drop=True)
        labels_matrix = labels_matrix[non_zero_rows]

        # one-hot -> 单个类别下标
        label_indices = np.argmax(labels_matrix, axis=1)

        # 是否合并样本太少的类别
        if to_merge and min_count > 1:
            counts = (labels_matrix == 1).sum(axis=0)
            to_merge_idx = [i for i, c in enumerate(counts) if c < min_count]
            remain_idx = [i for i in range(len(self.relation_fields)) if i not in to_merge_idx]

            merged_label_name = "_".join([self.relation_fields[i] for i in to_merge_idx]) \
                                or "Merged_Minority_Relation_Classes"

            self.final_fields = [self.relation_fields[i] for i in remain_idx] + [merged_label_name]

            # 映射老 index → 新 index
            old_to_new = {}
            for i in range(len(self.relation_fields)):
                if i in remain_idx:
                    old_to_new[i] = remain_idx.index(i)
                else:
                    old_to_new[i] = len(remain_idx)

            new_label_indices = [old_to_new[idx] for idx in label_indices]
            self.labels = new_label_indices
        else:
            self.final_fields = self.relation_fields
            self.labels = label_indices.tolist()

        # 把 ASN1/ASN2 保存下来，后面 __getitem__ 用
        self.asn1_list = self.df["ASN1"].astype(int).tolist()
        self.asn2_list = self.df["ASN2"].astype(int).tolist()

    def __len__(self):
        return len(self.asn1_list)

    def __getitem__(self, idx):
        asn1 = int(self.asn1_list[idx])
        asn2 = int(self.asn2_list[idx])
        label = int(self.labels[idx])

        if self.embedding_loader is not None:
            emb1 = self.embedding_loader.get_embedding(asn1)
            emb2 = self.embedding_loader.get_embedding(asn2)
            # # 你可以在这里选择：
            # # 1) 返回 (emb1, emb2, label)
            # # 2) 或者 cat 在一起: torch.cat([emb1, emb2]), label
            # # 下面采用第一种，由模型自己决定如何组合
            # return emb1, emb2, label
            
            # 拼成 (2, D)，DataLoader 后就是 (B, 2, D)
            pair_emb = torch.stack([emb1, emb2], dim=0)
            return pair_emb, label
        else:
            return asn1, asn2, label

    def get_label_map(self):
        """编号到关系名"""
        return {i: name for i, name in enumerate(self.final_fields)}

    def get_label_count(self):
        """每个关系类别的样本数"""
        from collections import Counter
        return dict(Counter(self.labels))



class ASEmbeddingLoader:
    def __init__(self, filepath, device="cpu"):
        """
        :param filepath: 路径到embedding txt文件
        :param device: 'cpu' 或 'cuda'
        """
        self.asn_to_embedding = {}
        self.device = device
        self._load_embeddings(filepath)
        
    def _load_embeddings(self, filepath):
        with open(filepath, 'r', encoding='utf8') as f:
            lines = f.readlines()
        header = lines[0].strip().split(",")
        # 数据部分
        for line in lines[1:]:
            vals = line.strip().split(",")
            asn = int(vals[0])
            emb = [float(x) for x in vals[1:]]
            # 转成torch tensor，指定device
            self.asn_to_embedding[asn] = torch.tensor(emb, dtype=torch.float32, device=self.device)
    
    def __contains__(self, asn):
        """便于判断ASN在不在embedding表里"""
        return asn in self.asn_to_embedding

    def get_embedding(self, asn):
        """
        :param asn: ASN (int或str，并自动转换)
        :return: 该ASN的embedding (tensor), 若不存在则抛KeyError
        """
        asn = int(asn)
        return self.asn_to_embedding[asn]
    
    def get_batch(self, asn_list):
        """
        :param asn_list: ASN列表
        :return: (N, emb_dim) 的 tensor, 没有的ASN抛出KeyError
        """
        emb_list = [self.get_embedding(asn) for asn in asn_list]
        return torch.stack(emb_list, dim=0)
    


    

# # ======= 用法示例 =======
# from torch.utils.data import DataLoader
# ds = ASCategoryDataset('/Users/ldd/Desktop/embed/node_features.csv', category='industry', min_count=500, to_merge=True)

# # 构建 DataLoader
# loader = DataLoader(ds, batch_size=4, shuffle=False)

# # 取出第一个batch
# first_batch = next(iter(loader))
# asn_batch, label_batch = first_batch

# print("ASN batch:", asn_batch)
# print("Label batch:", label_batch)
# print("类别映射：", ds.get_label_map())



# # 假设你的txt文件名为 "as_embedding.txt"
# loader = ASEmbeddingLoader("/Users/ldd/Desktop/embed/dataset/node2vec-embeddings16-10-100.txt", device="cpu")
# asn = 3356
# emb = loader.get_embedding(asn)   # 得到该AS的16维向量

# batch_emb = loader.get_batch([3356, 6939, 1299])  # (3, 16) tensor
# print(batch_emb)

# # 判断AS是否在embedding表里
# if 174 in loader:
#     print("ASN 174 有embedding")