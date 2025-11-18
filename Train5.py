# as_embedding_pretrain.py
# -*- coding: utf-8 -*-
import os
import math
import random
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


########################################
# 1. 一些工具函数
########################################

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


########################################
# 2. 数据读取与预处理
########################################

def load_features(feat_path):
    """
    读取 feat.txt
    格式示例：
    ASN, emb1, emb2, emb3, ..., embK
    6939, 0.5802919, 1497, 0.39, ...
    返回：
      - asn_list: [asn1, asn2, ...]
      - F: np.array, shape (num_asn, K) 原始特征（缺失已用 -1）
      - M: np.array, shape (num_asn, K) 缺失掩码 (1 表示缺失, 原值为 -1)
    """
    asn_list = []
    feats = []

    with open(feat_path, 'r') as f:
        header = f.readline()  # 跳过表头
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            asn = int(parts[0].strip())
            values = [float(x.strip()) for x in parts[1:]]
            asn_list.append(asn)
            feats.append(values)

    F = np.array(feats, dtype=np.float32)  # (N, K)
    # 构造缺失掩码 M：原始值为 -1 的位置 => 1
    M = (F == -1.0).astype(np.float32)
    return asn_list, F, M


def load_paths(path_file):
    """
    读取 AS-PATH 文本，每行： "3356 6939 15169 15169"
    返回：paths: List[List[int]]
    """
    paths = []
    with open(path_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                asns = [int(p) for p in parts]
            except ValueError:
                continue
            paths.append(asns)
    return paths


def build_vocab_and_filter(paths, feat_asns_set):
    """
    根据路径和 feature 中出现的 AS 构建词表。
    仅保留在 feat 里出现过的 AS，其他路径中的 AS 以及整条路径都删除。
    返回：
      - filtered_paths: 过滤过的路径（所有 AS 均在 feat_asns_set 中）
      - asn2id: {asn: idx}
      - id2asn: list[idx] = asn
    额外会创建特殊 token：
      - [PAD] = 0
      - [CLS] = 1
      - [SEP] = 2
      - [MASK] = 3
    """
    special_tokens = ['[PAD]', '[CLS]', '[SEP]', '[MASK]']
    PAD_ID, CLS_ID, SEP_ID, MASK_ID = 0, 1, 2, 3

    filtered_paths = []
    for p in paths:
        if all(a in feat_asns_set for a in p):
            filtered_paths.append(p)

    # 从过滤后的路径中构建 AS 集合
    as_set = set()
    for p in filtered_paths:
        as_set.update(p)

    # 只保留 feat 里的 AS
    as_set = as_set & feat_asns_set

    # 构建词表
    asn2id = {}
    id2asn = []

    # 先放特殊 token
    for i, tok in enumerate(special_tokens):
        asn2id[tok] = i
        id2asn.append(tok)

    # 接着放真实 AS
    for asn in sorted(as_set):
        idx = len(id2asn)
        asn2id[asn] = idx
        id2asn.append(asn)

    # 映射过滤路径到 id
    mapped_paths = []
    for p in filtered_paths:
        mapped = [asn2id[a] for a in p if a in asn2id]  # 理论上全在
        if len(mapped) > 0:
            mapped_paths.append(mapped)

    return mapped_paths, asn2id, id2asn, PAD_ID, CLS_ID, SEP_ID, MASK_ID


def build_feature_matrix(asn_list, F, M, asn2id):
    """
    将 feature 对齐到 vocab 上。
    输入：
      - asn_list: feature 文件里的 AS 顺序
      - F: (N, K) 原始特征
      - M: (N, K) 缺失掩码
      - asn2id: vocab 映射
    输出：
      - F_vocab: (V, K) 按词表顺序的特征（对特殊 token 行置 0）
      - M_vocab: (V, K) 按词表顺序的缺失掩码
    说明：
      - [PAD], [CLS], [SEP], [MASK] 这些特殊 token 没有真实特征，用 0 和 0 mask。
    """
    vocab_size = len(asn2id)
    K = F.shape[1]
    F_vocab = np.zeros((vocab_size, K), dtype=np.float32)
    M_vocab = np.zeros((vocab_size, K), dtype=np.float32)

    asn2row = {asn: i for i, asn in enumerate(asn_list)}

    for asn, idx in asn2id.items():
        if isinstance(asn, str):
            # 特殊 token
            continue
        if asn in asn2row:
            row = asn2row[asn]
            F_vocab[idx] = F[row]
            M_vocab[idx] = M[row]
        else:
            # 理论上不会出现，因为我们已经过滤了
            pass

    return F_vocab, M_vocab


########################################
# 3. Dataset：MAP + MFR + NSP 任务
########################################

class ASPDataset(Dataset):
    """
    为 MAP + MFR + NSP 提供样本。
    每个 __getitem__ 返回一个 NSP 样本：
      - input_ids: [CLS] A [SEP] B [SEP] （已做 mask 替换）
      - token_type_ids: segment id (0/1)
      - attention_mask: 1 for real, 0 for pad
      - map_labels: MLM/MAP 的目标 AS id（非 mask 位置为 -100）
      - feat_targets: 对被 mask 位置的特征重建目标 F' (内容特征+缺失掩码)，否则为 0
      - feat_mask: 对应位置是否参与 feature 重建 loss (1/0)
      - nsp_label: 1 表示 B 是 A 的真实后续，0 表示负样本
    """

    def __init__(
        self,
        paths,
        F_vocab,
        M_vocab,
        PAD_ID,
        CLS_ID,
        SEP_ID,
        MASK_ID,
        max_len=128,
        mask_prob=0.15
    ):
        self.paths = paths
        self.F_vocab = F_vocab       # (V, K)
        self.M_vocab = M_vocab       # (V, K)
        self.PAD_ID = PAD_ID
        self.CLS_ID = CLS_ID
        self.SEP_ID = SEP_ID
        self.MASK_ID = MASK_ID
        self.max_len = max_len
        self.mask_prob = mask_prob

        self.vocab_size = F_vocab.shape[0]
        self.K = F_vocab.shape[1]

        # 预先构造 F' = concat(F_processed, M)
        F_processed = F_vocab.copy()
        # 原始特征的 -1 替换为 0
        F_processed[F_processed == -1.0] = 0.0
        self.F_prime = np.concatenate([F_processed, M_vocab], axis=1)  # (V, 2K)

        # 每条路径会用来构造 NSP 正/负样本
        self.num_paths = len(self.paths)

    def __len__(self):
        # 每条路径我们构造两个样本（一个正，一个负）也可以，
        # 这里先简单按路径数
        return self.num_paths

    def _random_segment_split(self, path):
        """
        从真实路径中随机切一刀: A, B
        """
        if len(path) < 2:
            # 不足以切，直接整个当 A
            return path, []
        # 切点在 [1, len-1] 之间
        cut = random.randint(1, len(path) - 1)
        A = path[:cut]
        B = path[cut:]
        return A, B

    def _random_path(self, exclude_idx=None):
        """
        随机取一条路径，排除某个 index
        """
        while True:
            ridx = random.randint(0, self.num_paths - 1)
            if exclude_idx is None or ridx != exclude_idx:
                return self.paths[ridx]

    def _apply_mask(self, tokens):
        """
        对 tokens (不含 [CLS]/[SEP]) 做 BERT 风格的 mask，
        返回：
          - masked_tokens
          - map_labels: 预测目标，非 mask 位置为 -100
          - feat_mask: 对置 mask 的位置，1 表示参与 MFR
        """
        masked_tokens = tokens.copy()
        map_labels = np.full(len(tokens), -100, dtype=np.int64)
        feat_mask = np.zeros(len(tokens), dtype=np.float32)

        for i in range(len(tokens)):
            if random.random() < self.mask_prob:
                original_id = tokens[i]
                map_labels[i] = original_id
                feat_mask[i] = 1.0

                r = random.random()
                if r < 0.8:
                    # 80% 替换成 [MASK]
                    masked_tokens[i] = self.MASK_ID
                elif r < 0.9:
                    # 10% 替换成随机 token
                    masked_tokens[i] = random.randint(4, self.vocab_size - 1)
                else:
                    # 10% 保持原样
                    masked_tokens[i] = original_id

        return masked_tokens, map_labels, feat_mask

    def __getitem__(self, idx):
        # 1. 取一条真实路径
        path = self.paths[idx]

        # 2. 随机切 A, B 构造 NSP 正样本
        A, B = self._random_segment_split(path)

        # 如果 B 为空，则用随机 B 做负样本
        if len(B) == 0:
            # 直接当作负样本
            A = path
            B = self._random_path(exclude_idx=idx)
            nsp_label = 0
        else:
            # 50% 概率变成负样本
            if random.random() < 0.5:
                B = self._random_path(exclude_idx=idx)
                nsp_label = 0
            else:
                nsp_label = 1

        # 3. 截断到 max_len（要留出 [CLS], 两个 [SEP]）
        max_content_len = self.max_len - 3
        # 简单做法：A 和 B 总长度超过则截断 B
        if len(A) + len(B) > max_content_len:
            # 优先保留 A 的前半，截 B
            keep_B = max_content_len - len(A)
            if keep_B < 0:
                # A 也太长了，截 A
                A = A[:max_content_len]
                B = []
            else:
                B = B[:keep_B]

        # 4. 对 A、B 分别做 mask
        A_masked, A_map_labels, A_feat_mask = self._apply_mask(A)
        B_masked, B_map_labels, B_feat_mask = self._apply_mask(B)

        # 5. 组装 [CLS] A [SEP] B [SEP]
        tokens = [self.CLS_ID] + A_masked + [self.SEP_ID] + B_masked + [self.SEP_ID]
        seg_ids = [0] + [0] * len(A_masked) + [0] + [1] * len(B_masked) + [1]

        # 对 MAP/MFR 目标，CLS/SEP 不预测 => label 为 -100, feat_mask 为 0
        map_labels = [-100] + list(A_map_labels) + [-100] + list(B_map_labels) + [-100]
        feat_mask = [0.0] + list(A_feat_mask) + [0.0] + list(B_feat_mask) + [0.0]

        # 6. padding
        attn_mask = [1] * len(tokens)
        pad_len = self.max_len - len(tokens)
        if pad_len > 0:
            tokens += [self.PAD_ID] * pad_len
            seg_ids += [0] * pad_len
            attn_mask += [0] * pad_len
            map_labels += [-100] * pad_len
            feat_mask += [0.0] * pad_len
        else:
            tokens = tokens[:self.max_len]
            seg_ids = seg_ids[:self.max_len]
            attn_mask = attn_mask[:self.max_len]
            map_labels = map_labels[:self.max_len]
            feat_mask = feat_mask[:self.max_len]

        # 7. 准备 feature targets
        # 对于 map_labels != -100 的位置，我们需要预测 F'(token_id)
        # 否则 target 为 0
        feat_targets = np.zeros((self.max_len, self.F_prime.shape[1]), dtype=np.float32)
        for i in range(self.max_len):
            if map_labels[i] != -100:
                as_id = map_labels[i]  # 真实 AS id
                feat_targets[i] = self.F_prime[as_id]

        batch = {
            "input_ids": torch.LongTensor(tokens),
            "token_type_ids": torch.LongTensor(seg_ids),
            "attention_mask": torch.LongTensor(attn_mask),
            "map_labels": torch.LongTensor(map_labels),
            "feat_targets": torch.FloatTensor(feat_targets),
            "feat_mask": torch.FloatTensor(feat_mask),
            "nsp_label": torch.LongTensor([nsp_label]),
        }
        return batch


########################################
# 4. 模型定义
########################################

class FeatureMLP(nn.Module):
    """
    2 层 MLP，将 F' (2K) 投影到 d
    """
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class ASTransformerEncoder(nn.Module):
    """
    BERT-like Transformer Encoder + heads for:
      - MAP (Masked AS Prediction)
      - MFR (Masked Feature Reconstruction)
      - NSP (Path Continuation)
    """

    def __init__(
        self,
        vocab_size,
        d_model=128,
        n_heads=4,
        num_layers=4,
        max_len=128,
        feat_dim=16,      # 原始 K
        feat_hidden=256,  # MLP hidden
        dropout=0.1,
        use_gate=True
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.feat_dim = feat_dim
        self.use_gate = use_gate

        # Embedding
        self.id_embeddings = nn.Embedding(vocab_size, d_model)
        self.pos_embeddings = nn.Embedding(max_len, d_model)
        self.seg_embeddings = nn.Embedding(2, d_model)  # segment A/B

        # Feature projection
        # 输入 F' 为 2K 维
        self.feat_mlp = FeatureMLP(in_dim=2 * feat_dim, hidden_dim=feat_hidden, out_dim=d_model, dropout=dropout)

        # Gate: 根据 miss_ratio 或 M 来决定权重，这里用 miss_ratio 标量
        if use_gate:
            self.gate_w = nn.Linear(1, 1)  # 输入 miss_ratio (1-dim) -> scalar
        else:
            self.gate_w = None

        self.layer_norm = nn.LayerNorm(d_model)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Heads
        # MAP head
        self.map_head = nn.Linear(d_model, vocab_size)

        # MFR head: 回归 F' (2K)
        self.mfr_head = nn.Linear(d_model, 2 * feat_dim)

        # NSP head
        self.nsp_head = nn.Linear(d_model, 1)

    def forward(
        self,
        input_ids,
        token_type_ids,
        attention_mask,
        F_prime,   # (V, 2K)
        M_vocab,   # (V, K) 用于计算 miss_ratio
        output_hidden_states=False,
    ):
        """
        input_ids: (B, L)
        token_type_ids: (B, L)
        attention_mask: (B, L) 1/0
        F_prime: (V, 2K)
        M_vocab: (V, K)
        """
        B, L = input_ids.shape

        # 1. ID embedding
        x_id = self.id_embeddings(input_ids)  # (B, L, d)

        # 2. Feature embedding + gate
        #   对每个 token 查 F'，再用 MLP
        #   input_ids 中的特殊 token 在 F_prime 里是 0
        feat_in = F_prime[input_ids]  # (B, L, 2K)
        e_feat = self.feat_mlp(feat_in)  # (B, L, d)

        if self.use_gate:
            # miss_ratio: (#miss) / K, 从 M_vocab 查
            # 先算 for all vocab: miss_ratio_v = sum(M_vocab[v]) / K
            # 为了高效，预先在外部算好 miss_ratio_vocab 也可以，这里演示动态计算方式：
            with torch.no_grad():
                # 这里使用 CPU 的 numpy 会慢一些，如果需要高性能可在主程序中预先算好 miss_ratio_vocab
                pass

            # 简单起见，在模型初始化时，不预先存 miss_ratio，而是由外部传入也可以。
            # 但是这里我们演示一种“在线算”的方法：直接基于 F_prime/M_vocab
            # 为了示范，这里临时在 forward 里算 miss_ratio_vocab，并缓存在 buffer 里。
            if not hasattr(self, "miss_ratio_vocab"):
                # M_vocab: (V, K)
                M_tensor = torch.from_numpy(M_vocab).to(input_ids.device)  # (V, K)
                miss_ratio_vocab = M_tensor.mean(dim=1, keepdim=True)  # (V, 1)
                self.register_buffer("miss_ratio_vocab", miss_ratio_vocab)
            miss_ratio = self.miss_ratio_vocab[input_ids]  # (B, L, 1)
            gate = torch.sigmoid(self.gate_w(miss_ratio))  # (B, L, 1)
            e_feat = gate * e_feat
        else:
            # 不使用 gate，直接用 e_feat
            pass

        # 3. 位置 + segment embedding
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        x_pos = self.pos_embeddings(positions)
        x_seg = self.seg_embeddings(token_type_ids)

        x = x_id + e_feat + x_pos + x_seg
        x = self.layer_norm(x)

        # 4. Transformer Encoder
        # attention_mask: 1 for real, 0 for pad
        # nn.TransformerEncoder 使用 src_key_padding_mask: (B, L) True for pad
        src_key_padding_mask = (attention_mask == 0)
        # 也可以转换为 bool
        x_enc = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        # 5. Heads
        map_logits = self.map_head(x_enc)          # (B, L, V)
        mfr_out = self.mfr_head(x_enc)            # (B, L, 2K)
        cls_repr = x_enc[:, 0, :]                 # (B, d)
        nsp_logits = self.nsp_head(cls_repr)      # (B, 1)

        outputs = {
            "last_hidden_state": x_enc,
            "map_logits": map_logits,
            "mfr_out": mfr_out,
            "nsp_logits": nsp_logits,
        }
        if output_hidden_states:
            outputs["cls_repr"] = cls_repr
        return outputs

    def get_static_embeddings(self, F_prime, M_vocab):
        """
        方案一：静态 Embedding。
        不经过 Transformer，只用 AS 自身参数：
          emb_static[i] = LayerNorm(E_id[i] + g_i * e_feat[i])
        输入：
          - F_prime: (V, 2K)
          - M_vocab: (V, K)
        返回：
          - emb_static: (V, d)
        """
        device = next(self.parameters()).device
        vocab_size = self.vocab_size
        d_model = self.d_model
        K = M_vocab.shape[1]

        # 1. ID embedding
        ids = torch.arange(vocab_size, device=device, dtype=torch.long)
        E_id = self.id_embeddings(ids)  # (V, d)

        # 2. Feature projection
        F_prime_t = torch.from_numpy(F_prime).to(device)  # (V, 2K)
        e_feat = self.feat_mlp(F_prime_t)  # (V, d)

        # 3. Gate
        if self.use_gate:
            M_tensor = torch.from_numpy(M_vocab).to(device)  # (V, K)
            miss_ratio_vocab = M_tensor.mean(dim=1, keepdim=True)  # (V, 1)
            gate = torch.sigmoid(self.gate_w(miss_ratio_vocab))    # (V, 1)
            e_feat = gate * e_feat

        # 4. LayerNorm
        emb_static = self.layer_norm(E_id + e_feat)  # (V, d)
        return emb_static.detach().cpu().numpy()


########################################
# 5. 训练 & 导出 embedding
########################################

def train(
    model,
    train_loader,
    F_prime,
    M_vocab,
    device,
    num_epochs=5,
    lr=1e-4,
    lambda_map=1.0,
    lambda_mfr=1.0,
    lambda_nsp=0.5
):
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # 可以根据需要加 scheduler，这里先简单化

    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    mse_loss_fn = nn.MSELoss(reduction='none')
    bce_loss_fn = nn.BCEWithLogitsLoss()

    F_prime_t = torch.from_numpy(F_prime).to(device)  # (V, 2K)
    M_vocab_np = M_vocab  # 保留 numpy 给 model.forward 使用（内部会转 tensor）

    for epoch in range(num_epochs):
        total_loss = 0.0
        total_map_loss = 0.0
        total_mfr_loss = 0.0
        total_nsp_loss = 0.0
        steps = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)           # (B, L)
            token_type_ids = batch["token_type_ids"].to(device) # (B, L)
            attention_mask = batch["attention_mask"].to(device) # (B, L)
            map_labels = batch["map_labels"].to(device)         # (B, L)
            feat_targets = batch["feat_targets"].to(device)     # (B, L, 2K)
            feat_mask = batch["feat_mask"].to(device)           # (B, L)
            nsp_label = batch["nsp_label"].to(device)           # (B, 1)

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                F_prime=F_prime_t,
                M_vocab=M_vocab_np,
            )

            map_logits = outputs["map_logits"]       # (B, L, V)
            mfr_out = outputs["mfr_out"]             # (B, L, 2K)
            nsp_logits = outputs["nsp_logits"]       # (B, 1)

            B, L, V = map_logits.shape
            _, _, feat_dim2 = mfr_out.shape  # 2K

            # MAP loss
            map_loss = ce_loss_fn(
                map_logits.view(-1, V),
                map_labels.view(-1)
            )

            # MFR loss：对 feat_mask==1 的位置做 MSE
            mfr_loss_all = mse_loss_fn(mfr_out, feat_targets)  # (B, L, 2K)
            # broadcast feat_mask: (B, L) -> (B, L, 1)
            feat_mask_exp = feat_mask.unsqueeze(-1)            # (B, L, 1)
            mfr_loss_all = mfr_loss_all * feat_mask_exp
            # 只在 mask 位置归一化
            denom = feat_mask_exp.sum() * feat_dim2 + 1e-8
            mfr_loss = mfr_loss_all.sum() / denom

            # NSP loss
            nsp_loss = bce_loss_fn(
                nsp_logits.view(-1),
                nsp_label.view(-1).float()
            )

            loss = lambda_map * map_loss + lambda_mfr * mfr_loss + lambda_nsp * nsp_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_map_loss += map_loss.item()
            total_mfr_loss += mfr_loss.item()
            total_nsp_loss += nsp_loss.item()
            steps += 1

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"loss={total_loss/steps:.4f}, "
              f"map={total_map_loss/steps:.4f}, "
              f"mfr={total_mfr_loss/steps:.4f}, "
              f"nsp={total_nsp_loss/steps:.4f}")


def export_static_embeddings(model, F_prime, M_vocab, id2asn, save_path):
    """
    导出静态 embedding：
      - 文件格式：asn, e1, e2, ..., ed
    """
    emb_static = model.get_static_embeddings(F_prime, M_vocab)  # (V, d)
    vocab_size, d = emb_static.shape

    with open(save_path, "w") as f:
        header = ["ASN"] + [f"dim{i}" for i in range(d)]
        f.write(",".join(header) + "\n")
        for idx in range(vocab_size):
            asn = id2asn[idx]
            if isinstance(asn, str):
                # 特殊 token 可以选择跳过或写负值
                continue
            vec = emb_static[idx]
            line = [str(asn)] + [f"{v:.6f}" for v in vec.tolist()]
            f.write(",".join(line) + "\n")
    print(f"Static embeddings saved to {save_path}")


def export_contextual_embeddings(
    model,
    paths,
    asn2id,
    id2asn,
    F_prime,
    M_vocab,
    PAD_ID,
    CLS_ID,
    SEP_ID,
    MASK_ID,
    max_len,
    batch_size,
    device,
    save_path,
):
    """
    方案二：上下文平均 embedding。
    思路：
      - 再扫一遍所有路径，不做 mask，直接前向 Transformer 得到 H_i
      - 对于每个出现的 AS，累加其 hidden state 并计数，最后做平均
    """
    model.to(device)
    model.eval()

    vocab_size = len(id2asn)
    d_model = model.d_model

    sum_emb = torch.zeros((vocab_size, d_model), device=device)
    cnt = torch.zeros(vocab_size, device=device)

    F_prime_t = torch.from_numpy(F_prime).to(device)
    M_vocab_np = M_vocab

    # 为了简单，构造一个临时 Dataset 只输出单条路径的全序列（不做 NSP、mask）
    class SimplePathDataset(Dataset):
        def __init__(self, paths, max_len):
            self.paths = paths
            self.max_len = max_len

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            path = self.paths[idx]
            tokens = [CLS_ID] + path[: self.max_len - 2] + [SEP_ID]
            seg_ids = [0] * len(tokens)
            attn_mask = [1] * len(tokens)
            pad_len = self.max_len - len(tokens)
            if pad_len > 0:
                tokens += [PAD_ID] * pad_len
                seg_ids += [0] * pad_len
                attn_mask += [0] * pad_len

            return {
                "input_ids": torch.LongTensor(tokens),
                "token_type_ids": torch.LongTensor(seg_ids),
                "attention_mask": torch.LongTensor(attn_mask),
            }

    dataset = SimplePathDataset(paths, max_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)           # (B, L)
            token_type_ids = batch["token_type_ids"].to(device) # (B, L)
            attention_mask = batch["attention_mask"].to(device) # (B, L)

            outputs = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                F_prime=F_prime_t,
                M_vocab=M_vocab_np,
                output_hidden_states=True,
            )
            hidden = outputs["last_hidden_state"]  # (B, L, d)
            B, L, d = hidden.shape

            # 遍历每个位置非 PAD/非特殊 token（可以根据需要过滤 [CLS]/[SEP]）
            for b in range(B):
                for i in range(L):
                    token_id = input_ids[b, i].item()
                    if token_id in [PAD_ID, CLS_ID, SEP_ID, MASK_ID]:
                        continue
                    sum_emb[token_id] += hidden[b, i]
                    cnt[token_id] += 1

    # 计算平均
    cnt = cnt.clamp(min=1.0).unsqueeze(-1)  # (V,1)
    avg_emb = sum_emb / cnt  # (V, d_model)

    avg_emb = avg_emb.cpu().numpy()

    with open(save_path, "w") as f:
        header = ["ASN"] + [f"dim{i}" for i in range(d_model)]
        f.write(",".join(header) + "\n")
        for idx in range(vocab_size):
            asn = id2asn[idx]
            if isinstance(asn, str):
                continue
            vec = avg_emb[idx]
            line = [str(asn)] + [f"{v:.6f}" for v in vec.tolist()]
            f.write(",".join(line) + "\n")
    print(f"Contextual mean embeddings saved to {save_path}")


########################################
# 6. main 函数：串起来
########################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_file", type=str, default="AS_PATH.txt",
                        help="AS-PATH 文件，每行一个路径")
    parser.add_argument("--feat_file", type=str, default="feature_embeddings.txt",
                        help="feature 文件，第一列 ASN，后面是特征")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--feat_hidden", type=int, default=256)
    parser.add_argument("--lambda_map", type=float, default=1.0)
    parser.add_argument("--lambda_mfr", type=float, default=1.0)
    parser.add_argument("--lambda_nsp", type=float, default=0.5)
    parser.add_argument("--static_emb_out", type=str, default="emb_static.csv")
    parser.add_argument("--ctx_emb_out", type=str, default="emb_contextual.csv")
    args = parser.parse_args()

    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1. 读 feature
    asn_list, F, M = load_features(args.feat_file)
    print("Loaded features for", len(asn_list), "ASNs, feat_dim =", F.shape[1])
    feat_asns_set = set(asn_list)

    # 2. 读路径
    paths = load_paths(args.path_file)
    print("Loaded", len(paths), "paths")

    # 3. 构建 vocab & 过滤
    paths, asn2id, id2asn, PAD_ID, CLS_ID, SEP_ID, MASK_ID = build_vocab_and_filter(
        paths, feat_asns_set
    )
    print("After filtering, #paths =", len(paths), ", vocab_size =", len(asn2id))

    # 4. 对齐特征到 vocab
    F_vocab, M_vocab = build_feature_matrix(asn_list, F, M, asn2id)
    K = F_vocab.shape[1]
    print("Aligned feature matrix to vocab, K =", K)

    # 5. 构造 Dataset & DataLoader
    dataset = ASPDataset(
        paths=paths,
        F_vocab=F_vocab,
        M_vocab=M_vocab,
        PAD_ID=PAD_ID,
        CLS_ID=CLS_ID,
        SEP_ID=SEP_ID,
        MASK_ID=MASK_ID,
        max_len=args.max_len,
        mask_prob=0.15,
    )
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 6. 初始化模型
    model = ASTransformerEncoder(
        vocab_size=len(asn2id),
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        max_len=args.max_len,
        feat_dim=K,
        feat_hidden=args.feat_hidden,
        dropout=0.1,
        use_gate=True,
    )

    # 7. 训练
    train(
        model=model,
        train_loader=train_loader,
        F_prime=dataset.F_prime,  # (V, 2K)
        M_vocab=M_vocab,
        device=device,
        num_epochs=args.epochs,
        lr=args.lr,
        lambda_map=args.lambda_map,
        lambda_mfr=args.lambda_mfr,
        lambda_nsp=args.lambda_nsp,
    )

    # 8. 导出静态 embedding
    export_static_embeddings(
        model=model,
        F_prime=dataset.F_prime,
        M_vocab=M_vocab,
        id2asn=id2asn,
        save_path=args.static_emb_out,
    )

    # 9. 导出上下文平均 embedding
    export_contextual_embeddings(
        model=model,
        paths=paths,
        asn2id=asn2id,
        id2asn=id2asn,
        F_prime=dataset.F_prime,
        M_vocab=M_vocab,
        PAD_ID=PAD_ID,
        CLS_ID=CLS_ID,
        SEP_ID=SEP_ID,
        MASK_ID=MASK_ID,
        max_len=args.max_len,
        batch_size=args.batch_size,
        device=device,
        save_path=args.ctx_emb_out,
    )


if __name__ == "__main__":
    main()