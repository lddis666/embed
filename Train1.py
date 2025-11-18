import os
import math
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

######################################################################
# 配置区域（根据需要修改）
######################################################################

AS_PATH_FILE = "as_paths.txt"
AS_FEATURE_FILE = "as_features.txt"

# 最大路径长度
L_MAX = 128

# 模型超参
D_MODEL = 128
NUM_LAYERS = 4
NUM_HEADS = 4
FFN_HIDDEN = 4 * D_MODEL
DROPOUT = 0.1

# 预训练任务权重
LAMBDA_MAP = 1.0
LAMBDA_MFR = 1.0
LAMBDA_NSP = 0.5

# 训练超参
BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 3e-4
WARMUP_STEPS = 1000
MAX_STEPS = 100000  # 用于余弦退火，可根据数据规模设置

# NSP 负样本比例
NSP_NEG_PROB = 0.5

# 随机种子
RANDOM_SEED = 42

######################################################################
# 工具函数：设备 & 随机性
######################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(RANDOM_SEED)

######################################################################
# 数据加载：AS Feature
######################################################################


def load_as_features(feature_file):
    """
    读取 AS feature 文件 (CSV):
    ASN, f1, f2, ..., fK
    返回:
        asn_list: [asn1, asn2, ...] (按出现顺序)
        F: torch.FloatTensor, shape = [num_asn, K]
        M: torch.FloatTensor, shape = [num_asn, K] (1 表示缺失)
    """
    import csv

    asn_list = []
    feats = []

    with open(feature_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)  # 丢掉表头
        for row in reader:
            if len(row) < 2:
                continue
            asn = int(row[0])
            vals = []
            for v in row[1:]:
                if v == "" or v.lower() == "nan":
                    vals.append(-1.0)
                else:
                    vals.append(float(v))
            asn_list.append(asn)
            feats.append(vals)

    F = torch.tensor(feats, dtype=torch.float)
    # 缺失掩码：原始值为 -1 的位置记为 1
    M = (F == -1).float()
    # 将 -1 替换为 0
    F_processed = F.clone()
    F_processed[F_processed == -1] = 0.0

    return asn_list, F_processed, M


######################################################################
# 数据加载：AS PATH
######################################################################


def load_as_paths(path_file):
    """
    读取 AS PATH 文件，每行一个路径，空格分隔
    返回：list of list[int]
    """
    paths = []
    with open(path_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tokens = line.split()
            try:
                asns = [int(t) for t in tokens]
            except ValueError:
                continue
            paths.append(asns)
    return paths


######################################################################
# 词表 & 特征矩阵构建
######################################################################


class VocabAndFeatureBuilder:
    """
    1. 根据 AS-PATH 构建词表
    2. 和 feature 对齐，删除在 feature 中不存在的 ASN
    3. 生成最终的:
       - asn2idx / idx2asn
       - F': concat(F_processed, M)
    """

    def __init__(self, paths, asn_list, F, M):
        """
        paths: list[list[int]]
        asn_list: list[int] (feature 文件中的 ASN 顺序)
        F: [num_asn, K]
        M: [num_asn, K]
        """
        self.raw_paths = paths
        self.asn_list = asn_list
        self.F = F
        self.M = M

        self._build()

    def _build(self):
        # 1. 统计 PATH 中出现过的 ASN
        as_in_paths = set()
        for p in self.raw_paths:
            as_in_paths.update(p)

        # 2. feature 文件中的 ASN 到 index 的映射
        feat_asn2idx = {asn: i for i, asn in enumerate(self.asn_list)}

        # 3. 只保留既在 PATH 又在 feature 中的 ASN
        valid_asn = sorted([asn for asn in as_in_paths if asn in feat_asn2idx])

        # 0/1/2 预留给：PAD, [CLS], [SEP], [MASK]
        special_tokens = ["[PAD]", "[CLS]", "[SEP]", "[MASK]"]
        self.PAD_ID = 0
        self.CLS_ID = 1
        self.SEP_ID = 2
        self.MASK_ID = 3

        self.asn2idx = {}
        self.idx2asn = {}

        # 特殊 token
        for i, t in enumerate(special_tokens):
            self.asn2idx[t] = i
            self.idx2asn[i] = t

        # 普通 ASN
        offset = len(special_tokens)
        for i, asn in enumerate(valid_asn):
            idx = offset + i
            self.asn2idx[asn] = idx
            self.idx2asn[idx] = asn

        self.vocab_size = len(self.asn2idx)

        # 构造 F'，只针对普通 ASN
        K = self.F.size(1)
        num_as = len(valid_asn)
        F_selected = torch.zeros((num_as, K), dtype=torch.float)
        M_selected = torch.zeros((num_as, K), dtype=torch.float)

        for i, asn in enumerate(valid_asn):
            j = feat_asn2idx[asn]
            F_selected[i] = self.F[j]
            M_selected[i] = self.M[j]

        # F' = concat(F_processed, M)
        self.F_concat = torch.cat([F_selected, M_selected], dim=1)  # [num_as, 2K]
        self.K = K
        self.num_as = num_as

        # 对应的 miss_ratio
        miss_count = M_selected.sum(dim=1)  # [num_as]
        self.miss_ratio = miss_count / float(self.K)

        # 4. 过滤 PATH：删除不存在于 asn2idx 的 ASN（或整条 PATH）
        filtered_paths = []
        for p in self.raw_paths:
            new_p = [asn for asn in p if asn in self.asn2idx]
            if len(new_p) >= 2:  # 太短的路径跳过
                filtered_paths.append(new_p)
        self.paths = filtered_paths

    def asn_to_id(self, asn):
        return self.asn2idx.get(asn, None)

    def path_to_ids(self, path):
        ids = []
        for asn in path:
            if asn in self.asn2idx:
                ids.append(self.asn2idx[asn])
        return ids


######################################################################
# Dataset：包含 MAP + MFR + NSP
######################################################################


class ASPTDataset(Dataset):
    def __init__(self, builder: VocabAndFeatureBuilder, max_len=L_MAX, nsp_neg_prob=NSP_NEG_PROB):
        """
        为了简单，这里：
        - 对每条路径用于 MAP + MFR
        - 同时在 __getitem__ 中构造 NSP 样本 (A, B, is_next)
        """
        self.builder = builder
        self.paths = builder.paths
        self.max_len = max_len
        self.nsp_neg_prob = nsp_neg_prob

    def __len__(self):
        return len(self.paths)

    def _truncate_and_pad(self, ids):
        # 截断
        ids = ids[: self.max_len]
        # padding
        pad_len = self.max_len - len(ids)
        if pad_len > 0:
            ids = ids + [self.builder.PAD_ID] * pad_len
        return ids

    def _mask_for_map(self, token_ids):
        """
        BERT 风格的 mask：
        - 15% 的 token 作为预测目标
        - 其中 80% -> [MASK], 10% -> random token, 10% -> keep
        返回：
            masked_ids, mask_positions, mask_labels
        """
        input_ids = list(token_ids)
        labels = [-100] * len(input_ids)  # -100 表示不计算 loss

        # 选择可 mask 的位置（排除 PAD/CLS/SEP/MASK，实际这里只是 PATH 中的 ASN）
        cand_indices = [i for i, tid in enumerate(input_ids) if tid not in
                        [self.builder.PAD_ID, self.builder.CLS_ID, self.builder.SEP_ID, self.builder.MASK_ID]]

        num_to_mask = max(1, int(len(cand_indices) * 0.15))
        random.shuffle(cand_indices)
        mask_pos = cand_indices[:num_to_mask]

        for pos in mask_pos:
            original_id = input_ids[pos]
            prob = random.random()
            labels[pos] = original_id
            if prob < 0.8:
                # 80% 替换为 [MASK]
                input_ids[pos] = self.builder.MASK_ID
            elif prob < 0.9:
                # 10% 随机替换
                rand_id = random.randint(0, self.builder.vocab_size - 1)
                input_ids[pos] = rand_id
            else:
                # 10% 保持不变
                pass
        return input_ids, labels

    def _build_nsp_example(self, path_ids):
        """
        构造 NSP 样本：
        - 正样本：A,B 来自同一条路径
        - 负样本：B 来自另一条路径或打乱
        返回：
            input_ids, segment_ids, is_next_label
        格式：[CLS] A [SEP] B [SEP]
        """
        if len(path_ids) < 2:
            # 太短，直接当整条 A，B 空
            A = path_ids
            B = []
            is_next = 1
        else:
            cut = random.randint(1, len(path_ids) - 1)
            A = path_ids[:cut]
            B_real = path_ids[cut:]
            if random.random() < self.nsp_neg_prob:
                # 负样本
                # 从其他路径中选一条 B
                while True:
                    other = random.choice(self.paths)
                    if other is not path_ids:
                        break
                other_ids = [self.builder.asn2idx[a] for a in other]
                # 也可随机子片段
                if len(other_ids) > 1:
                    s = random.randint(0, len(other_ids) - 1)
                    e = random.randint(s + 1, len(other_ids))
                    B = other_ids[s:e]
                else:
                    B = other_ids
                is_next = 0
            else:
                # 正样本
                B = B_real
                is_next = 1

        # 添加 [CLS] [SEP] 标记
        input_ids = [self.builder.CLS_ID] + A + [self.builder.SEP_ID] + B + [self.builder.SEP_ID]
        segment_ids = []
        # segment: A=0, B=1
        # [CLS]
        segment_ids.append(0)
        # A
        segment_ids.extend([0] * len(A))
        # [SEP]
        segment_ids.append(0)
        # B
        segment_ids.extend([1] * len(B))
        # [SEP]
        segment_ids.append(1)

        # 截断 + padding
        input_ids = input_ids[: self.max_len]
        segment_ids = segment_ids[: self.max_len]
        pad_len = self.max_len - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.builder.PAD_ID] * pad_len
            segment_ids = segment_ids + [0] * pad_len

        return input_ids, segment_ids, is_next

    def __getitem__(self, idx):
        path = self.paths[idx]
        # path 中的 ASN -> id
        path_ids = [self.builder.asn2idx[a] for a in path]

        # 1) MAP + MFR 的序列（只用一段）
        # 这里简单起见，不加 CLS/SEP，只是单序列，后续模型里也可以用同一输入
        token_ids = self._truncate_and_pad(path_ids)
        masked_ids, map_labels = self._mask_for_map(token_ids)

        # 2) NSP 序列
        nsp_input_ids, nsp_segment_ids, is_next = self._build_nsp_example(path_ids)

        return {
            "map_input_ids": torch.tensor(masked_ids, dtype=torch.long),
            "map_labels": torch.tensor(map_labels, dtype=torch.long),
            "nsp_input_ids": torch.tensor(nsp_input_ids, dtype=torch.long),
            "nsp_segment_ids": torch.tensor(nsp_segment_ids, dtype=torch.long),
            "nsp_label": torch.tensor(is_next, dtype=torch.long),
        }


######################################################################
# 模型定义
######################################################################


class FeatureGateModule(nn.Module):
    """
    MLP_feat: F' -> R^d
    gate: miss_ratio -> scalar in (0,1)
    """

    def __init__(self, input_dim, d_model):
        super().__init__()
        hidden_dim = max(64, input_dim * 2)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )
        self.gate_w = nn.Linear(1, 1)  # 使用 miss_ratio (标量) 决定 gate

    def forward(self, F_concat, miss_ratio):
        """
        F_concat: [num_as, 2K]
        miss_ratio: [num_as]
        """
        e_feat = self.mlp(F_concat)  # [num_as, d]
        g = torch.sigmoid(self.gate_w(miss_ratio.view(-1, 1))).view(-1)  # [num_as]
        return e_feat, g


class ASBertModel(nn.Module):
    def __init__(self, vocab_size, F_concat, miss_ratio, pad_id, cls_id, sep_id, mask_id,
                 d_model=D_MODEL, num_layers=NUM_LAYERS, num_heads=NUM_HEADS,
                 ffn_hidden=FFN_HIDDEN, dropout=DROPOUT):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.cls_id = cls_id
        self.sep_id = sep_id
        self.mask_id = mask_id

        self.d_model = d_model

        # ID embedding
        self.tok_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)

        # Position embedding
        self.max_len = L_MAX
        self.pos_embed = nn.Embedding(self.max_len, d_model)

        # Segment embedding
        self.seg_embed = nn.Embedding(2, d_model)

        # Feature + Gate
        self.register_buffer("F_concat", F_concat)      # [num_as, 2K]
        self.register_buffer("miss_ratio", miss_ratio)  # [num_as]
        self.num_special = 4  # PAD,CLS,SEP,MASK
        self.feature_gate = FeatureGateModule(F_concat.size(1), d_model)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_hidden,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # MAP head (vocab prediction)
        self.map_head = nn.Linear(d_model, vocab_size)

        # MFR head (feature reconstruction)
        self.mfr_head = nn.Linear(d_model, F_concat.size(1))  # -> 2K

        # NSP head
        self.nsp_head = nn.Linear(d_model, 2)

    def _build_input_embedding(self, input_ids, segment_ids=None):
        """
        input_ids: [B,L]
        segment_ids: [B,L] or None (None -> all 0)
        """
        B, L = input_ids.size()
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        if segment_ids is None:
            segment_ids = torch.zeros_like(input_ids)

        tok_emb = self.tok_embed(input_ids)  # [B,L,d]
        pos_emb = self.pos_embed(positions)
        seg_emb = self.seg_embed(segment_ids)

        # 构造 feature embedding + gate
        # 特殊 token（0..3）没有对应的 F_concat，简单做法：其 feature embedding = 0
        F_concat_full = torch.zeros(B, L, self.F_concat.size(1), device=input_ids.device)
        miss_ratio_full = torch.zeros(B, L, device=input_ids.device)

        # 对普通 ASN 填充
        # 普通 asn 的索引范围 [num_special, vocab_size)
        # 转成 mask
        normal_mask = (input_ids >= self.num_special)
        if normal_mask.any():
            # 映射到 feature 索引：asn_id - num_special
            asn_idx = (input_ids - self.num_special).clamp(min=0)
            F_concat_full[normal_mask] = self.F_concat[asn_idx[normal_mask]]
            miss_ratio_full[normal_mask] = self.miss_ratio[asn_idx[normal_mask]]

        # 计算 e_feat & gate
        # 展平计算，再 reshape
        F_concat_flat = F_concat_full.view(-1, F_concat_full.size(-1))
        miss_ratio_flat = miss_ratio_full.view(-1)
        e_feat_flat, g_flat = self.feature_gate(F_concat_flat, miss_ratio_flat)
        e_feat = e_feat_flat.view(B, L, -1)
        g = g_flat.view(B, L, 1)

        # x_i = e_id + g * e_feat + e_pos + e_seg
        x = tok_emb + g * e_feat + pos_emb + seg_emb
        x = self.layer_norm(x)
        x = self.dropout(x)
        return x

    def forward(self,
                map_input_ids, map_labels,
                nsp_input_ids, nsp_segment_ids, nsp_label):
        """
        map_input_ids: [B,L]
        map_labels: [B,L] (masked positions -> real id, others -100)
        nsp_input_ids: [B,L]
        nsp_segment_ids: [B,L]
        nsp_label: [B]
        """
        B, L = map_input_ids.size()

        # 1) MAP + MFR 使用同一输入（这里为了简单，先只对 map_input_ids 做一次 Transformer）
        map_embed = self._build_input_embedding(map_input_ids, None)
        # padding mask: True 表示要被 mask（即 ignore）
        pad_mask_map = (map_input_ids == self.pad_id)  # [B,L]
        map_hidden = self.encoder(map_embed, src_key_padding_mask=pad_mask_map)  # [B,L,d]

        # 任务一：MAP
        map_logits = self.map_head(map_hidden)  # [B,L,vocab]
        map_loss = F.cross_entropy(
            map_logits.view(-1, self.vocab_size),
            map_labels.view(-1),
            ignore_index=-100,
        )

        # 任务二：MFR
        # 重构 F_concat。这里只重构整向量，简单用 MSE。
        F_hat = self.mfr_head(map_hidden)  # [B,L,2K]
        # 构造真实 F_concat
        with torch.no_grad():
            # 同 _build_input_embedding 过程
            F_concat_full = torch.zeros(B, L, self.F_concat.size(1), device=map_input_ids.device)
            normal_mask = (map_input_ids >= self.num_special)
            if normal_mask.any():
                asn_idx = (map_input_ids - self.num_special).clamp(min=0)
                F_concat_full[normal_mask] = self.F_concat[asn_idx[normal_mask]]
        # 只对非 PAD token 做 MFR loss
        valid_mfr_mask = (map_input_ids != self.pad_id).unsqueeze(-1)  # [B,L,1]
        diff = (F_hat - F_concat_full) * valid_mfr_mask
        mfr_loss = (diff ** 2).sum() / (valid_mfr_mask.sum() * diff.size(-1) + 1e-8)

        # 2) NSP：重新构造 NSP 输入的 embedding + encoder
        nsp_embed = self._build_input_embedding(nsp_input_ids, nsp_segment_ids)
        pad_mask_nsp = (nsp_input_ids == self.pad_id)
        nsp_hidden = self.encoder(nsp_embed, src_key_padding_mask=pad_mask_nsp)  # [B,L,d]
        # 取 [CLS] 位置 hidden
        cls_pos = (nsp_input_ids == self.cls_id).nonzero(as_tuple=False)
        # 如果每个样本只有一个 CLS，且在开头，可以简化：
        # cls_hidden = nsp_hidden[:, 0, :]
        # 这里更通用一些
        cls_hidden = []
        for b in range(B):
            # 找到该样本的 CLS 位置
            pos = (nsp_input_ids[b] == self.cls_id).nonzero(as_tuple=False)
            if len(pos) == 0:
                cls_hidden.append(nsp_hidden[b, 0])  # fallback
            else:
                cls_hidden.append(nsp_hidden[b, pos[0].item()])
        cls_hidden = torch.stack(cls_hidden, dim=0)  # [B,d]

        nsp_logits = self.nsp_head(cls_hidden)  # [B,2]
        nsp_loss = F.cross_entropy(nsp_logits, nsp_label)

        total_loss = LAMBDA_MAP * map_loss + LAMBDA_MFR * mfr_loss + LAMBDA_NSP * nsp_loss

        return {
            "loss": total_loss,
            "map_loss": map_loss,
            "mfr_loss": mfr_loss,
            "nsp_loss": nsp_loss,
        }

    ##################################################################
    # 导出 static embedding
    ##################################################################
    def export_static_embeddings(self):
        """
        方案一：静态 embedding
        emb_static[i] = LayerNorm( E_id[i] + g_i * e_feat[i] )
        对所有 ASN（不含特殊 token）
        返回:
            emb_static: [vocab_size, d]，其中 0..3 为特殊 token 的 embedding
        """
        with torch.no_grad():
            # 先获得 ID embedding 表
            tok_weights = self.tok_embed.weight.clone()  # [vocab_size,d]

            # 普通 ASN 部分
            num_normal = self.F_concat.size(0)
            F_concat = self.F_concat
            miss_ratio = self.miss_ratio

            e_feat, g = self.feature_gate(F_concat, miss_ratio)  # [num_as,d], [num_as]
            g = g.unsqueeze(-1)  # [num_as,1]
            # 普通 asn id 从 num_special 开始
            static_emb = tok_weights.clone()
            static_emb[self.num_special:self.num_special + num_normal] = \
                self.layer_norm(static_emb[self.num_special:self.num_special + num_normal] + g * e_feat)

        return static_emb  # [vocab_size,d]

    ##################################################################
    # 导出 contextual mean embedding
    ##################################################################
    def export_contextual_embeddings(self, dataset: ASPTDataset, batch_size=64):
        """
        方案二：contextual mean embedding
        再扫一遍语料（这里用 dataset.paths），对每条 PATH 前向一次，不做 mask，
        累加每个 ASN 的 H_i，然后取平均。
        返回：
            emb_ctx: [vocab_size, d]
        """
        self.eval()
        with torch.no_grad():
            vocab_size = self.vocab_size
            d = self.d_model
            sum_emb = torch.zeros(vocab_size, d, device=device)
            cnt = torch.zeros(vocab_size, device=device)

            # 简单使用原始路径（不做 NSP 格式）
            all_paths = dataset.paths
            # 作一个 DataLoader
            def collate_fn(batch_paths):
                max_len = max(len(p) for p in batch_paths)
                max_len = min(max_len, self.max_len)
                input_ids = []
                for p in batch_paths:
                    ids = [self.cls_id] + [dataset.builder.asn2idx[a] for a in p][: (max_len - 2)] + [self.sep_id]
                    pad_len = max_len - len(ids)
                    ids = ids + [dataset.builder.PAD_ID] * pad_len
                    input_ids.append(ids)
                input_ids = torch.tensor(input_ids, dtype=torch.long)
                return input_ids

            tmp_loader = DataLoader(all_paths, batch_size=batch_size,
                                    shuffle=False, collate_fn=collate_fn)

            for batch_paths_ids in tmp_loader:
                batch_paths_ids = batch_paths_ids.to(device)
                B, L = batch_paths_ids.size()
                seg_ids = torch.zeros_like(batch_paths_ids)
                emb = self._build_input_embedding(batch_paths_ids, seg_ids)
                pad_mask = (batch_paths_ids == self.pad_id)
                hidden = self.encoder(emb, src_key_padding_mask=pad_mask)  # [B,L,d]

                for b in range(B):
                    for t in range(L):
                        asn_id = batch_paths_ids[b, t].item()
                        if asn_id in [self.pad_id, self.cls_id, self.sep_id, self.mask_id]:
                            continue
                        sum_emb[asn_id] += hidden[b, t]
                        cnt[asn_id] += 1

            # 平均
            cnt = cnt.clamp(min=1.0).unsqueeze(-1)
            emb_ctx = sum_emb / cnt
        return emb_ctx  # [vocab_size,d]


######################################################################
# 学习率调度（warmup + cosine）
######################################################################


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, max_steps, min_lr=1e-7):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        self.step_num = 0
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def step(self):
        self.step_num += 1
        for i, group in enumerate(self.optimizer.param_groups):
            base_lr = self.base