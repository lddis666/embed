import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from collections import Counter


from torch.utils.data import Dataset
import random
from itertools import combinations


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
    # "industry": [
    #     # "Agriculture, Mining & Utilities",
    #     ["ASDB_C1L1_Agriculture, Mining, and Refineries (Farming, Greenhouses, Mining, Forestry, and Animal Farming)",
    #     "ASDB_C1L1_Utilities (Excluding Internet Service)"],

    #     # "Information & Media",
    #     ["ASDB_C1L1_Computer and Information Technology",
    #     "ASDB_C1L1_Media, Publishing, and Broadcasting"],

    #     # "Construction, Manufacturing & Freight",
    #     ["ASDB_C1L1_Construction and Real Estate",
    #     "ASDB_C1L1_Manufacturing",
    #     "ASDB_C1L1_Freight, Shipment, and Postal Services"],

    #     # "Government, Education & Nonprofits",
    #     ["ASDB_C1L1_Government and Public Administration",
    #     "ASDB_C1L1_Education and Research",
    #     "ASDB_C1L1_Community Groups and Nonprofits"],

    #     # "Finance, Retail & Service",
    #     ["ASDB_C1L1_Finance and Insurance",
    #     "ASDB_C1L1_Retail Stores, Wholesale, and E-commerce Sites",
    #     "ASDB_C1L1_Service"],

    #     # "Health, Travel & Entertainment",
    #     ["ASDB_C1L1_Health Care Services",
    #     "ASDB_C1L1_Travel and Accommodation",
    #     "ASDB_C1L1_Museums, Libraries, and Entertainment"],

    #     "ASDB_C1L1_Other"
    # ]


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
        
        self.df[self.fields] = self.df[self.fields].fillna(0)
        labels_matrix = self.df[self.fields].values.astype(float)
        



        # 删除没有类别的AS
        non_zero_rows = np.where(labels_matrix.sum(axis=1) != 0)[0]
        if len(non_zero_rows) < len(self.df):
            print(f"Warning: {len(self.df)-len(non_zero_rows)} rows dropped due to all-zero labels.")
        self.df = self.df.iloc[non_zero_rows].reset_index(drop=True)
        self.ASNs = [self.ASNs[i] for i in non_zero_rows]
        labels_matrix = labels_matrix[non_zero_rows]

        unique_vals = np.unique(labels_matrix)
        print("标签矩阵取值集合:", unique_vals)
        assert set(unique_vals).issubset({0., 1.}), "标签列出现非 0/1 数值"

        
        # 找出每行对应的类别下标
        label_indices = np.argmax(labels_matrix, axis=1)  # one-hot只有一个为1


        row_sums = labels_matrix.sum(axis=1)
        multi_label_rows = np.where(row_sums > 1)[0]
        if len(multi_label_rows) > 0:
            print(f"Warning: {len(multi_label_rows)} rows have multiple labels; only the first one (argmax) is kept.")


        # ====== 合并前：打印类别映射及样本数 ======
        pre_merge_counts = Counter(label_indices)
        print("=== 合并前（过滤/删全零之后）的类别分布 ===")
        for idx, cnt in sorted(pre_merge_counts.items()):
            print(f"  原始类别索引 {idx}（字段名 {self.fields[idx]}）: 样本数 {cnt}")
        print(f"  总样本数: {sum(pre_merge_counts.values())}")
        # ====================================

        # 合并类别（数量太小的放一起）
        if to_merge and min_count > 1:
            # 统计每一列（类别）的正样本数量
            counts = (labels_matrix == 1 ).sum(axis=0)
            to_merge_idx = [i for i, c in enumerate(counts) if ("None" not in self.fields[i]) and c < min_count]
            remain_idx = [i for i in range(len(self.fields)) if i not in to_merge_idx]
            merged_label_name = '_'.join([self.fields[i] for i in to_merge_idx]) or 'Merged_Minority_Classes'
            # 新类别名顺序: 保留的+合并的
            self.final_fields = [self.fields[i] for i in remain_idx] + [merged_label_name]

            print("=== 依据 min_count 的原始列统计（合并前）===")
            for i, c in enumerate(counts):
                print(f"  列索引 {i}（字段名 {self.fields[i]}）: 正样本数 {c}")
            print(f"  将被合并的列索引: {to_merge_idx}")
            print(f"  保留为独立类别的列索引: {remain_idx}")
            print(f"  合并后类别数: {len(self.final_fields)}")
            print(f"  合并后类别名顺序: {self.final_fields}")


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

            print("将被合并的列索引:", to_merge_idx)
            print("保留为独立类别的列索引:", remain_idx)
            print("合并后类别数:", len(remain_idx) + (1 if to_merge_idx else 0))

            post_merge_counts = Counter(self.labels)
            print("=== 合并后（最终）的类别分布 ===")
            for idx, cnt in sorted(post_merge_counts.items()):
                print(f"  新类别索引 {idx}（字段名 {self.final_fields[idx]}）: 样本数 {cnt}")
            print(f"  总样本数: {sum(post_merge_counts.values())}")
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



# class ASCategoryDataset(Dataset):
#     def __init__(self, csv_path, category, min_count=1, to_merge=False,
#                  embedding_loader=None, filter_asns=None):
#         self.df = pd.read_csv(csv_path)
#         print(len(self.df))
#         self.category = category

#         # ---------------- 解析 CATEGORIES[category] ----------------
#         raw_cfg = CATEGORIES[category]  # 里面既可能有 str 也可能有 list[str]

#         field_groups = []      # 每个元素是一个 list[str]，表示要合并成一个类别的列
#         final_fields = []      # 每个最终类别的名字（你可以自定义命名规则）
#         original_fields = []   # 扁平化之后的所有原始列名

#         for item in raw_cfg:
#             if isinstance(item, str):
#                 # 单独一个列就是一个类别
#                 field_groups.append([item])
#                 final_fields.append(item)
#                 original_fields.append(item)
#             elif isinstance(item, (list, tuple)):
#                 # 多个列合并成一个类别
#                 group = list(item)
#                 field_groups.append(group)
#                 # 给这个合并后的类别起个名字，你可以改成自己喜欢的规则
#                 merged_name = "+".join(group)
#                 final_fields.append(merged_name)
#                 original_fields.extend(group)
#             else:
#                 raise ValueError(f"Unsupported category config type: {type(item)} in CATEGORIES[{category}]")

#         self.field_groups = field_groups          # 例如 [[col1], [col2], [col3, col4]]
#         self.final_fields = final_fields          # 例如 ["col1", "col2", "col3+col4"]
#         self.original_fields = original_fields    # 例如 ["col1", "col2", "col3", "col4"]

#         self.embedding_loader = embedding_loader
#         self.ASNs = self.df['ASN'].astype(int).tolist()

#         # ---------------- 下面保持你原来的过滤逻辑 ----------------
#         if self.embedding_loader is not None:
#             has_embedding = [asn for asn in self.ASNs if asn in self.embedding_loader]
#             self.df = self.df[self.df['ASN'].astype(int).isin(has_embedding)].reset_index(drop=True)
#             print(len(self.df))
#             self.ASNs = self.df['ASN'].astype(int).tolist()

#         if filter_asns is not None:
#             filter_asns = set(filter_asns)
#             before = len(self.df)
#             self.df = self.df[self.df['ASN'].astype(int).isin(filter_asns)].reset_index(drop=True)
#             print(len(filter_asns))
#             print(f"Filtered by user list: {before} → {len(self.df)}")
#             self.ASNs = self.df['ASN'].astype(int).tolist()

#         # --------- 从 df 中取出原始列（扁平化后）---------
#         labels_matrix_raw = self.df[self.original_fields].values.astype(float)

#         # 删除没有任何标签的行（全零）
#         non_zero_rows = np.where(labels_matrix_raw.sum(axis=1) != 0)[0]
#         if len(non_zero_rows) < len(self.df):
#             print(f"Warning: {len(self.df)-len(non_zero_rows)} rows dropped due to all-zero labels.")
#         self.df = self.df.iloc[non_zero_rows].reset_index(drop=True)
#         self.ASNs = [self.ASNs[i] for i in non_zero_rows]
#         labels_matrix_raw = labels_matrix_raw[non_zero_rows]

#         # --------- 把原始列合并成最终的类别列 ---------
#         # 目标：得到形状为 (N, num_final_classes) 的矩阵 labels_matrix
#         num_samples = labels_matrix_raw.shape[0]
#         num_classes = len(self.field_groups)
#         labels_matrix = np.zeros((num_samples, num_classes), dtype=float)

#         # 建立 原始列名 -> 在 labels_matrix_raw 中的索引
#         col_to_idx = {name: i for i, name in enumerate(self.original_fields)}

#         for new_j, group in enumerate(self.field_groups):
#             idxs = [col_to_idx[col] for col in group]
#             # 这里的合并逻辑是「行上求和」，如果是 one-hot 列的话，相当于 group 中有任意一个为 1 就算这个新类为 1
#             # 你也可以改成 np.max(labels_matrix_raw[:, idxs], axis=1) 按逻辑“或”合并
#             group_vals = labels_matrix_raw[:, idxs].sum(axis=1)
#             # 如果你相信不会有多标签情况，这里可以转成 0/1
#             group_vals = (group_vals > 0).astype(float)
#             labels_matrix[:, new_j] = group_vals

#         # 删除合并后仍然是全零的行（比如原先只有被你删掉的列的）
#         non_zero_rows2 = np.where(labels_matrix.sum(axis=1) != 0)[0]
#         if len(non_zero_rows2) < len(self.df):
#             print(f"Warning: {len(self.df)-len(non_zero_rows2)} rows dropped after merging labels.")
#         self.df = self.df.iloc[non_zero_rows2].reset_index(drop=True)
#         self.ASNs = [self.ASNs[i] for i in non_zero_rows2]
#         labels_matrix = labels_matrix[non_zero_rows2]

#         # --------- 从 one-hot 得到类别下标 ---------
#         label_indices = np.argmax(labels_matrix, axis=1)  # 假设每行只有一个 1

#         # --------- 你原来 to_merge / min_count 的逻辑（可选） ---------
#         if to_merge and min_count > 1:
#             counts = (labels_matrix == 1).sum(axis=0)
#             # 注意: 现在 self.final_fields 是合并后类别的名字，其中不再有 "None" 这种；
#             # 如果还有需要过滤 "None" 的逻辑，你可以自行调整判断条件
#             to_merge_idx = [i for i, c in enumerate(counts)
#                             if ("None" not in self.final_fields[i]) and c < min_count]
#             remain_idx = [i for i in range(len(self.final_fields)) if i not in to_merge_idx]
#             merged_label_name = '_'.join([self.final_fields[i] for i in to_merge_idx]) or 'Merged_Minority_Classes'

#             new_final_fields = [self.final_fields[i] for i in remain_idx] + [merged_label_name]

#             old_to_new = {}
#             for i in range(len(self.final_fields)):
#                 if i in remain_idx:
#                     old_to_new[i] = remain_idx.index(i)
#                 else:
#                     old_to_new[i] = len(remain_idx)

#             new_label_indices = [old_to_new[idx] for idx in label_indices]

#             self.final_fields = new_final_fields
#             self.labels = new_label_indices
#         else:
#             # 如果不按 min_count 再合并，小类合并逻辑已经结束
#             self.labels = label_indices.tolist()
            




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

        row_sums = labels_matrix.sum(axis=1)
        multi_label_rows = np.where(row_sums > 1)[0]
        if len(multi_label_rows) > 0:
            print(f"Warning: {len(multi_label_rows)} rows have multiple labels; only the first one (argmax) is kept.")

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
    







class LinkPredictionDataset(Dataset):
    """
    用于图的 link prediction。
    输入：两个节点（或它们的 embedding）
    输出：是否存在边（1: 有边, 0: 无边）
    """
    def __init__(
        self,
        csv_path,
        src_col="src_id",
        dst_col="dst_id",
        embedding_loader=None,
        filter_nodes=None,
        num_pos_samples=None,   # 想要的正样本数量
        negative_ratio=1.0,     # 负样本 : 正样本
        undirected=False,
        seed=42,
    ):
        """
        :param csv_path: 边列表 CSV 路径, 例如:
                         src_id,dst_id
                         1,7843
                         1,11537
        :param src_col: 源点列名
        :param dst_col: 目标点列名
        :param embedding_loader: 可选，提供节点 embedding 的对象
                                 要支持: id in loader 以及 loader.get_embedding(id)
        :param filter_nodes: 只保留这些节点之间的关系 (list 或 set)，
                             负样本也只在该集合中采样
        :param num_pos_samples: 希望使用的“正样本”条数。
                                若为 None，则使用所有正边。
                                若大于可用正样本数，则自动降为最大可用数。
        :param negative_ratio: 每个正样本生成多少个负样本
        :param undirected: 是否视为无向图；True 时 (u,v) 和 (v,u) 视为同一条边
        :param seed: 随机种子
        """
        self.embedding_loader = embedding_loader
        self.undirected = undirected
        self.rng = random.Random(seed)

        df = pd.read_csv(csv_path)
        df[src_col] = df[src_col].astype(int)
        df[dst_col] = df[dst_col].astype(int)

        # label 名称映射：0 -> negative, 1 -> positive
        self.label_map = {
            0: "no_edge",
            1: "edge",
        }

        # 无向图则规范化 (u,v)，并去重
        if undirected:
            u = df[src_col].values
            v = df[dst_col].values
            u2 = np.minimum(u, v)
            v2 = np.maximum(u, v)
            df[src_col] = u2
            df[dst_col] = v2
            df = df.drop_duplicates(subset=[src_col, dst_col]).reset_index(drop=True)

        # 按 embedding_loader 过滤
        if self.embedding_loader is not None:
            def has_both_embeddings(row):
                return (int(row[src_col]) in self.embedding_loader) and \
                       (int(row[dst_col]) in self.embedding_loader)
            before = len(df)
            df = df[df.apply(has_both_embeddings, axis=1)].reset_index(drop=True)
            print(f"Filtered by embedding_loader: {before} → {len(df)} edges")

        # 按 filter_nodes 过滤
        if filter_nodes is not None:
            filter_nodes = set(int(x) for x in filter_nodes)
            before = len(df)
            mask = df[src_col].isin(filter_nodes) & df[dst_col].isin(filter_nodes)
            df = df[mask].reset_index(drop=True)
            print(f"Filtered by user node set: {before} → {len(df)} edges")

        # 所有正样本（边）
        all_pos_edges = list(zip(df[src_col].tolist(), df[dst_col].tolist()))
        all_pos_edges = [(int(u), int(v)) for u, v in all_pos_edges]
        total_pos = len(all_pos_edges)
        if total_pos == 0:
            raise ValueError("No positive edges left after filtering.")

        # 随机打乱所有正边
        self.rng.shuffle(all_pos_edges)

        # 决定实际使用的正样本数量
        if num_pos_samples is None or num_pos_samples <= 0:
            num_pos_samples = total_pos
        if num_pos_samples > total_pos:
            print(f"Requested num_pos_samples={num_pos_samples} > available={total_pos}, "
                  f"using {total_pos} instead.")
            num_pos_samples = total_pos

        pos_edges = all_pos_edges[:num_pos_samples]
        print(f"Use {len(pos_edges)} positive samples (from {total_pos} available).")

        # 用于快速判断是否有边
        if undirected:
            self.edge_set = set((min(u, v), max(u, v)) for u, v in all_pos_edges)
        else:
            self.edge_set = set(all_pos_edges)

        # 节点集合（负样本从这里采）
        if filter_nodes is not None:
            nodes = sorted(filter_nodes)
        else:
            nodes = sorted(set(df[src_col].tolist()) | set(df[dst_col].tolist()))
        self.nodes = [int(x) for x in nodes]

        # ======================
        # 负样本构造
        # ======================
        num_neg_target = int(len(pos_edges) * negative_ratio)

        N = len(self.nodes)
        if N < 2:
            raise ValueError("Not enough nodes to construct negative samples.")

        # 所有可能的候选边数量（不含自环）
        if undirected:
            max_pairs = N * (N - 1) // 2
        else:
            max_pairs = N * (N - 1)

        # 如果节点数不大，可以精确枚举所有对
        if max_pairs <= 2_000_000 and N <= 5000:
            if undirected:
                all_pairs = set((min(u, v), max(u, v)) for u, v in combinations(self.nodes, 2))
            else:
                all_pairs = set((u, v) for u in self.nodes for v in self.nodes if u != v)
            all_neg = list(all_pairs - self.edge_set)
            self.rng.shuffle(all_neg)
            neg_edges = all_neg[:num_neg_target]
        else:
            # 随机采样负样本
            neg_edges = []
            tried = 0
            max_try = num_neg_target * 20  # 防止死循环
            while len(neg_edges) < num_neg_target and tried < max_try:
                u = self.rng.choice(self.nodes)
                v = self.rng.choice(self.nodes)
                if u == v:
                    tried += 1
                    continue
                if undirected:
                    p = (min(u, v), max(u, v))
                else:
                    p = (u, v)
                if p in self.edge_set:
                    tried += 1
                    continue
                # 防止重复
                if p in neg_edges:
                    tried += 1
                    continue
                neg_edges.append(p)
                tried += 1

        print(f"Final: positive={len(pos_edges)}, negative={len(neg_edges)}")

        # ======================
        # 合并正负样本并打乱
        # ======================
        data_pairs = pos_edges + neg_edges
        labels = [1] * len(pos_edges) + [0] * len(neg_edges)

        idx = list(range(len(data_pairs)))
        self.rng.shuffle(idx)
        self.u_list = [int(data_pairs[i][0]) for i in idx]
        self.v_list = [int(data_pairs[i][1]) for i in idx]
        self.labels = [int(labels[i]) for i in idx]

    def __len__(self):
        return len(self.u_list)

    def __getitem__(self, idx):
        u = self.u_list[idx]
        v = self.v_list[idx]
        label = self.labels[idx]

        if self.embedding_loader is not None:
            emb_u = self.embedding_loader.get_embedding(u)
            emb_v = self.embedding_loader.get_embedding(v)
            pair_emb = torch.stack([emb_u, emb_v], dim=0)  # (2, D)
            return pair_emb, label
        else:
            return u, v, label

    def get_label_count(self):
        from collections import Counter
        return dict(Counter(self.labels))

    def get_label_map(self):
        """
        返回：编号到标签名的映射
        0 -> "no_edge"
        1 -> "edge"
        """
        return dict(self.label_map)

    

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
# emb = loader.get_embedding(asn)   # 得到该AS的向量

# batch_emb = loader.get_batch([3356, 6939, 1299])  # (3, 16) tensor
# print(batch_emb)

# # 判断AS是否在embedding表里
# if 174 in loader:
#     print("ASN 174 有embedding")