# as_embedding_pretrain.py
import os
import json
import math
import random
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

########################################
#           配置与超参数
########################################

class Config:
    # 文件路径
    as_path_file = "filtered_paths.txt"      # 每行: "3356 6939 15169"
    feature_file = "feature_embeddings.txt"          # CSV: "ASN, emb1, emb2, ..., embK"
    output_dir = "output"

    # 模型尺寸
    d_model = 128        # embedding 维度
    num_layers = 2       # Transformer 层数
    num_heads = 4        # Multi-head attention 头数
    d_ff = 256           # Transformer 内部 FFN 维度
    dropout = 0.1

    # 特征维度相关（会在读 feat.txt 时自动确定）
    # K: 原始连续特征个数
    # F': 2K: 特征 + 缺失mask

    # 路径处理
    max_seq_len = 32
    mask_prob = 0.3     # MAP/MFR 的 mask 比例

    # 训练相关
    batch_size = 512
    num_epochs = 60
    learning_rate = 3e-4
    weight_decay = 1e-3
    warmup_steps = 100

    # 多任务权重
    lambda_map = 2.0
    lambda_mfr = 0.0

    # 其它
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_name = "1129-map-with-feat"


cfg = Config()

os.makedirs(cfg.output_dir, exist_ok=True)
random.seed(cfg.seed)
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(cfg.seed)

########################################
#           数据读取与预处理
########################################

def load_features(feature_file: str):
    """
    读取特征文件 feat.txt
    格式示例：
    ASN, emb1, emb2, ..., embK
    6939, 0.5802919, 1497, 0.39, ...
    返回：
        asn_list: List[int]
        features: np.ndarray, shape (n_asn, K)
    """
    asn_list = []
    feat_list = []
    with open(feature_file, "r") as f:
        header = f.readline().strip().split(",")
        # 假设第一列是 ASN，其余是特征
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            asn = int(parts[0])
            vals = [float(x) for x in parts[1:]]
            asn_list.append(asn)
            feat_list.append(vals)
    features = np.array(feat_list, dtype=np.float32)
    return asn_list, features


def load_paths(path_file: str) -> List[List[int]]:
    """
    读取 AS-PATH，每行一个路径: "3356 6939 15169"
    返回：List[List[int]]，每条路径为 ASN 列表
    """
    paths = []
    with open(path_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                path = [int(x) for x in parts]
            except ValueError:
                continue
            if len(path) == 0:
                continue
            paths.append(path)
    return paths


def build_vocab_from_paths_and_features(paths: List[List[int]],
                                        feat_asn_list: List[int]) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    构建 ASN -> idx 的词表，只保留在 AS PATH 和 feature 中都出现的 ASN。
    返回：
        asn2idx: {asn: idx}, idx 从 0..V-1
        idx2asn: {idx: asn}
    """
    asn_in_paths = set()
    for p in paths:
        for a in p:
            asn_in_paths.add(a)
    asn_in_feat = set(feat_asn_list)
    valid_asn = sorted(list(asn_in_paths & asn_in_feat))

    asn2idx = {asn: i for i, asn in enumerate(valid_asn)}
    idx2asn = {i: asn for asn, i in asn2idx.items()}
    return asn2idx, idx2asn


def filter_paths_and_features(paths: List[List[int]],
                              asn2idx: Dict[int, int],
                              feat_asn_list: List[int],
                              feat_values: np.ndarray):
    """
    1) 过滤路径：如果一条路径中某个 ASN 不在 asn2idx 中，则丢掉整条路径
    2) 过滤特征：只保留 asn2idx 中的 ASN
    返回：
        filtered_paths_idx: List[List[int]]  路径中的 ASN 用 idx 表示
        feat_mat: np.ndarray, shape (|V|, K), 与 asn2idx 对齐
    """
    # 过滤路径
    filtered_paths_idx = []
    for p in paths:
        valid = True
        idx_path = []
        for a in p:
            if a not in asn2idx:
                valid = False
                break
            idx_path.append(asn2idx[a])
        if valid and len(idx_path) > 0:
            filtered_paths_idx.append(idx_path)

    # 过滤特征
    # 原始 feat_asn_list 有顺序，feat_values 与之对齐
    asn2featrow = {asn: i for i, asn in enumerate(feat_asn_list)}
    V = len(asn2idx)
    K = feat_values.shape[1]
    feat_mat = np.zeros((V, K), dtype=np.float32)
    for asn, idx in asn2idx.items():
        row = asn2featrow[asn]
        feat_mat[idx] = feat_values[row]
    return filtered_paths_idx, feat_mat


def build_feature_with_missing(feat_mat: np.ndarray):
    """
    对特征做缺失编码：
    - 原始缺失值用 -1 表示
    - 构造缺失掩码 M: M[i, k] = 1 if feat == -1 else 0
    - 将 feat 中的 -1 替换为 0
    - 拼接: F' = concat(feat_processed, M) -> shape (V, 2K)
    返回：
        F_prime: np.ndarray, shape (V, 2K)
        M: np.ndarray, shape (V, K)
        miss_ratio: np.ndarray, shape (V,)
    """
    V, K = feat_mat.shape
    M = (feat_mat == -1).astype(np.float32)
    feat_processed = feat_mat.copy()
    feat_processed[feat_processed == -1] = 0.0
    F_prime = np.concatenate([feat_processed, M], axis=1)
    miss_ratio = M.sum(axis=1) / float(K)
    return F_prime, M, miss_ratio


########################################
#             Dataset 定义
########################################

CLS_ID = 0
SEP_ID = 1
MASK_ID = 2
PAD_ID = 3
SPECIAL_TOKENS = {
    "CLS": CLS_ID,
    "SEP": SEP_ID,
    "MASK": MASK_ID,
    "PAD": PAD_ID,
}

def build_token_mapping(asn2idx: Dict[int, int]) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    为 ASN idx 建立 token idx：
    0..3 为特殊 token, 从 4 开始是 AS token
    返回：
        asnIdx2tokenIdx: {asn_idx: token_idx}
        tokenIdx2asnIdx: {token_idx: asn_idx} (特殊 token 的值设为 -1)
    """
    V = len(asn2idx)
    asnIdx2tokenIdx = {}
    tokenIdx2asnIdx = {}
    offset = len(SPECIAL_TOKENS)  # 4
    for asn, idx in asn2idx.items():
        token_idx = idx + offset
        asnIdx2tokenIdx[idx] = token_idx
        tokenIdx2asnIdx[token_idx] = idx

    # 特殊 token 映射到 -1
    for name, tid in SPECIAL_TOKENS.items():
        tokenIdx2asnIdx[tid] = -1

    return asnIdx2tokenIdx, tokenIdx2asnIdx


class ASPathDataset(Dataset):
    """
    用于 MAP + MFR 预训练的 Dataset。
    在 AS PATH 上做 token-level 的 mask（MAP + MFR）。
    """

    def __init__(self,
                 paths_idx: List[List[int]],
                 asnIdx2tokenIdx: Dict[int, int],
                 tokenIdx2asnIdx: Dict[int, int],
                 max_seq_len: int,
                 mask_prob: float,
                 F_prime: np.ndarray,
                 miss_ratio: np.ndarray,
                 for_inference: bool = False):
        """
        paths_idx: List[List[int]], 每个元素是 AS idx（0..V-1)
        asnIdx2tokenIdx: 将 AS idx 映射到 token idx (从4开始)
        F_prime: shape (V, 2K)  # 只对 AS idx 有意义
        miss_ratio: shape(V,)
        """
        self.paths = paths_idx
        self.asnIdx2tokenIdx = asnIdx2tokenIdx
        self.tokenIdx2asnIdx = tokenIdx2asnIdx
        self.max_seq_len = max_seq_len
        self.mask_prob = mask_prob
        self.F_prime = F_prime
        self.miss_ratio = miss_ratio
        self.for_inference = for_inference


    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        """
        返回：
            input_ids: [CLS] path [SEP] + padding
            token_type_ids: segment 0/1
            attention_mask: 1 for real tokens, 0 for pad
            map_labels: 对 MAP 的标签（同大小，非 mask 位置为 -100）
            mfr_targets: 对 MFR 的目标特征 F'（对无 AS / pad 位置填 0）
            mfr_mask:  对哪些位置参与 MFR loss（相当于 mask 的位置）
        """

        path = self.paths[idx]
        # 截断，让 [CLS] path [SEP] 总长不超过 max_seq_len
        max_len_for_path = self.max_seq_len - 2  # CLS + SEP
        if len(path) > max_len_for_path:
            path = path[:max_len_for_path]

        # 转换为 token idx
        def asnIdx_list_to_tokenIdx_list(asn_list):
            return [self.asnIdx2tokenIdx[a] for a in asn_list]
        

        path_tok = asnIdx_list_to_tokenIdx_list(path)
        tokens = [CLS_ID] + path_tok + [SEP_ID]
        seg_ids = [0] * len(tokens)  # 只有一段
        
        attention_mask = [1] * len(tokens)
        pad_len = self.max_seq_len - len(tokens)


        if pad_len > 0:
            tokens += [PAD_ID] * pad_len
            seg_ids += [0] * pad_len
            attention_mask += [0] * pad_len
        else:
            tokens = tokens[:self.max_seq_len]
            seg_ids = seg_ids[:self.max_seq_len]
            attention_mask = attention_mask[:self.max_seq_len]

        tokens = np.array(tokens, dtype=np.int64)
        seg_ids = np.array(seg_ids, dtype=np.int64)
        attention_mask = np.array(attention_mask, dtype=np.float32)


        if self.for_inference:
            # 推理/统计模式：不做 mask，不返回 label
            return {
                "input_ids": torch.from_numpy(tokens),
                "token_type_ids": torch.from_numpy(seg_ids),
                "attention_mask": torch.from_numpy(attention_mask),
            }

        # 准备 MAP + MFR 的 mask
        # 对真实 AS token（非特殊、非 pad）以 mask_prob 的概率进行 mask
        map_labels = np.full_like(tokens, fill_value=-100, dtype=np.int64)
        mfr_targets = np.zeros((self.max_seq_len, self.F_prime.shape[1]), dtype=np.float32)
        mfr_mask = np.zeros(self.max_seq_len, dtype=np.float32)

        for i in range(len(tokens)):
            tid = tokens[i]
            if tid in [CLS_ID, SEP_ID, PAD_ID, MASK_ID]:
                continue
            # 这个 tid 对应的是 AS token，参与 MLM
            if random.random() < self.mask_prob:
                # MAP 标签是 token 本身
                map_labels[i] = tid

                # MFR 目标特征取对应 AS 的 F'
                asn_idx = self.tokenIdx2asnIdx.get(tid, -1)
                if asn_idx >= 0:
                    mfr_targets[i] = self.F_prime[asn_idx]
                else:
                    mfr_targets[i] = 0.0
                mfr_mask[i] = 1.0

                # 按 BERT 策略：
                r = random.random()
                if r < 0.8:
                    tokens[i] = MASK_ID
                elif r < 0.9:
                    # 随机替换为其他 token
                    # 简化起见，从全部 token 范围选一个
                    tokens[i] = random.randint(4, 4 + len(self.asnIdx2tokenIdx) - 1)
                else:
                    # 保持原 token
                    pass

        return {
            "input_ids": torch.from_numpy(tokens),
            "token_type_ids": torch.from_numpy(seg_ids),
            "attention_mask": torch.from_numpy(attention_mask),
            "map_labels": torch.from_numpy(map_labels),
            "mfr_targets": torch.from_numpy(mfr_targets),
            "mfr_mask": torch.from_numpy(mfr_mask),
        }

       

########################################
#             模型定义
########################################

class FeatureMLP(nn.Module):
    """
    将 F' (2K) -> d_model 的 MLP
    """

    def __init__(self, in_dim, d_model, hidden_dim=None, dropout=0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = d_model * 2
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class GateModule(nn.Module):
    """
    根据 miss_ratio 或缺失模式决定 gate
    这里用简单版本：g_i = sigmoid(w * miss_ratio + b)
    """

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))
        self.b = nn.Parameter(torch.zeros(1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, miss_ratio: torch.Tensor):
        # miss_ratio: shape (V,) or (...,)
        # broadcast 到与 e_feat 相同的 batch 形状时使用
        g = self.sigmoid(self.w * miss_ratio + self.b)
        return g


class ASBertModel(nn.Module):
    """
    整体 BERT-like 模型：
    - token embedding (特殊 token + AS ID embedding)
    - position embedding
    - segment embedding
    - 特征 MLP + gate
    - Transformer encoder
    - 两个 heads: MAP(vocab), MFR(feature)
    """

    def __init__(self,
                 vocab_size,
                 d_model,
                 num_layers,
                 num_heads,
                 d_ff,
                 dropout,
                 F_prime: np.ndarray,
                 miss_ratio: np.ndarray,
                 tokenIdx2asnIdx: Dict[int, int]):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.tokenIdx2asnIdx = tokenIdx2asnIdx

        # 基础 embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(cfg.max_seq_len, d_model)
        self.seg_embedding = nn.Embedding(2, d_model)

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Feature MLP
        self.F_prime = nn.Parameter(torch.from_numpy(F_prime), requires_grad=False)  # (V, 2K)
        self.miss_ratio = nn.Parameter(torch.from_numpy(miss_ratio).float(), requires_grad=False)  # (V,)


        self.feature_mlp = FeatureMLP(F_prime.shape[1], d_model, hidden_dim=d_ff, dropout=dropout)
        self.gate = GateModule()

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # MAP head
        self.map_classifier = nn.Linear(d_model, vocab_size)

        # MFR head (regress to F' dim)
        self.mfr_head = nn.Linear(d_model, F_prime.shape[1])


        token_to_asn = torch.full((vocab_size,), -1, dtype=torch.long)
        for tid, asn_idx in tokenIdx2asnIdx.items():
            token_to_asn[tid] = asn_idx
        self.register_buffer("token_to_asn", token_to_asn)

    def get_as_static_embedding(self):
        """
        导出静态 embedding（不经过 Transformer）：
        emb_static[i] = LayerNorm( E_id[i_token] + g_i * e_feat[i] )
        注意：
        - 只对 AS token (非特殊 token) 导出
        - 返回: tensor (V, d_model) 和对应 ASN index 顺序
        """
        device = next(self.parameters()).device
        V = self.F_prime.size(0)
        # AS id 对应的 token idx: offset 4
        offset = len(SPECIAL_TOKENS)
        token_ids = torch.arange(offset, offset + V, dtype=torch.long, device=device)
        E_id = self.token_embedding(token_ids)  # (V, d_model)
        Fp = self.F_prime.to(device)            # (V, 2K)
        miss_r = self.miss_ratio.to(device)     # (V,)

        e_feat = self.feature_mlp(Fp)           # (V, d_model)
        g = self.gate(miss_r).unsqueeze(-1)     # (V, 1)
        emb = E_id + g * e_feat
        emb = self.layer_norm(emb)
        return emb  # (V, d_model)

    def forward(self,
                input_ids,
                token_type_ids,
                attention_mask,
                map_labels=None,
                mfr_targets=None,
                mfr_mask=None,):
        """
        input_ids: (B, L)
        token_type_ids: (B, L)
        attention_mask: (B, L)
        """

        device = input_ids.device
        B, L = input_ids.size()

        # 基础 embedding: token + pos + seg
        pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        tok_emb = self.token_embedding(input_ids)
        pos_emb = self.pos_embedding(pos_ids)
        seg_emb = self.seg_embedding(token_type_ids)

        x = tok_emb + pos_emb + seg_emb

        # 加入特征 embedding (对每个位置)
        # 对于非 AS token (特殊 token、PAD)，不加特征
        # 对 AS token: e_feat = MLP(F'[asn_idx]); g = gate(miss_ratio[asn_idx])
        # 需要从 token_idx -> asn_idx
        # asn_idx_tensor = torch.full_like(input_ids, fill_value=-1)
        # for tid, asn_idx in self.tokenIdx2asnIdx.items():
        #     # tid: token idx
        #     if asn_idx >= 0:
        #         asn_idx_tensor[input_ids == tid] = asn_idx

        asn_idx_tensor = self.token_to_asn[input_ids]  # (B, L), -1 表示非 AS token

        # mask: 是否是 AS token
        is_as_token = asn_idx_tensor >= 0
        if is_as_token.any():
            # 取出对应 AS 的 F'
            asn_idx_flat = asn_idx_tensor[is_as_token]
            Fp = self.F_prime[asn_idx_flat].to(device)         # (N_as, 2K)
            miss_r = self.miss_ratio[asn_idx_flat].to(device)  # (N_as,)

            e_feat = self.feature_mlp(Fp)                      # (N_as, d_model)
            g = self.gate(miss_r).unsqueeze(-1)                # (N_as, 1)
            e_feat = g * e_feat

            x = x.clone()
            x[is_as_token] = x[is_as_token] + e_feat

        x = self.layer_norm(x)
        x = self.dropout(x)

        # Transformer encoder
        # attention_mask: 1 for real, 0 for pad
        # PyTorch Transformer uses attn_mask or src_key_padding_mask
        src_key_padding_mask = (attention_mask == 0)  # (B, L) bool
        h = self.encoder(x, src_key_padding_mask=src_key_padding_mask)  # (B, L, d_model)

        # MAP 头 (token-level)
        map_logits = self.map_classifier(h)  # (B, L, vocab_size)

        # MFR 头 (token-level)
        feat_pred = self.mfr_head(h)        # (B, L, 2K)



        outputs = {
            "map_logits": map_logits,
            "feat_pred": feat_pred,
            "hidden_states": h,
        }

        # 如果传入了 label，就计算 loss
        total_loss = None
        if map_labels is not None:
            # MAP loss
            map_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            map_loss = map_loss_fct(
                map_logits.view(-1, self.vocab_size),
                map_labels.view(-1)
            )

            # MFR loss: 只在 mfr_mask == 1 的位置计算 L1/MSE 等
            # 这里用 MSE
            if mfr_targets is not None and mfr_mask is not None:
                mfr_mask = mfr_mask.unsqueeze(-1)  # (B, L, 1)
                mse = (feat_pred - mfr_targets.to(device)) ** 2
                mse = mse * mfr_mask.to(device)
                # 避免除 0
                denom = mfr_mask.sum() * feat_pred.size(-1) + 1e-8
                mfr_loss = mse.sum() / denom
            else:
                mfr_loss = torch.tensor(0.0, device=device)



            total_loss = (cfg.lambda_map * map_loss
                          + cfg.lambda_mfr * mfr_loss)




            # total_loss = cfg.lambda_map * map_loss

            outputs["loss"] = total_loss
            outputs["map_loss"] = map_loss
            outputs["mfr_loss"] = mfr_loss

        return outputs


########################################
#           训练工具：LR Scheduler
########################################

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    与 transformers 中类似的 warmup+线性下降调度器
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / max(1, num_warmup_steps)
        return max(
            0.0, float(num_training_steps - current_step) / max(1, num_training_steps - num_warmup_steps)
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


########################################
#           上下文平均 Embedding
########################################

def build_contextual_embeddings(model: ASBertModel,
                                dataloader: DataLoader,
                                tokenIdx2asnIdx: Dict[int, int]):
    """
    扫一遍语料，取得 Transformer 输出 H_i，对每个 AS 位置累加平均。
    """
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        # max token idx for AS
        # 找到所有 AS  token 用的 asn_idx 范围
        asn_indices = [a for a in tokenIdx2asnIdx.values() if a >= 0]
        V = max(asn_indices) + 1 if asn_indices else 0
        sum_emb = torch.zeros((V, model.d_model), device=device)
        cnt = torch.zeros(V, device=device)

        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # forward without loss
            out = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                map_labels=None,
                mfr_targets=None,
                mfr_mask=None,
            )
            h = out["hidden_states"]  

            B, L = input_ids.size()


            # 统计每个 AS 的平均 H
            for b in range(B):
                for t in range(L):
                    tid = input_ids[b, t].item()
                    if tid in [CLS_ID, SEP_ID, PAD_ID, MASK_ID]:
                        continue
                    asn_idx = tokenIdx2asnIdx.get(tid, -1)
                    if asn_idx < 0:
                        continue
                    sum_emb[asn_idx] += h[b, t]
                    cnt[asn_idx] += 1.0

        # 计算平均
        cnt_expand = cnt.unsqueeze(-1).clamp(min=1.0)
        emb_ctx = sum_emb / cnt_expand
        return emb_ctx


########################################
#                 Main
########################################

if __name__ == "__main__":
    # 1. 读取数据
    print("Loading features...")
    feat_asn_list, feat_values = load_features(cfg.feature_file)
    print("Loading paths...")
    raw_paths = load_paths(cfg.as_path_file)

    # 2. 构建词表，只保留同时在 PATH 和 feature 中出现的 ASN
    print("Building vocab...")
    asn2idx, idx2asn = build_vocab_from_paths_and_features(raw_paths, feat_asn_list)
    print(f"Num AS in vocab: {len(asn2idx)}")

    # 3. 过滤路径和特征
    print("Filtering paths and features...")
    paths_idx, feat_mat = filter_paths_and_features(raw_paths, asn2idx, feat_asn_list, feat_values)
    print(f"Num valid paths: {len(paths_idx)}")

    # 4. 特征缺失编码
    print("Building feature with missing mask...")
    F_prime, M, miss_ratio = build_feature_with_missing(feat_mat)
    V, twoK = F_prime.shape
    print(f"Feature matrix F' shape: {F_prime.shape}")

    # 5. 构建 token 映射
    asnIdx2tokenIdx, tokenIdx2asnIdx = build_token_mapping(asn2idx)
    vocab_size = len(tokenIdx2asnIdx)  # 包含特殊 token

    # 6. 构建 Dataset & DataLoader
    print("Building dataset...")
    dataset = ASPathDataset(paths_idx, asnIdx2tokenIdx, tokenIdx2asnIdx, cfg.max_seq_len, cfg.mask_prob, F_prime, miss_ratio)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)

    # 7. 构建模型
    print("Building model...")
    model = ASBertModel(
        vocab_size=vocab_size,
        d_model=cfg.d_model,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
        F_prime=F_prime,
        miss_ratio=miss_ratio,
        tokenIdx2asnIdx=tokenIdx2asnIdx,
    ).to(cfg.device)

    # 8. 优化器与 Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    num_training_steps = cfg.num_epochs * len(dataloader)
    scheduler = get_linear_schedule_with_warmup(optimizer, cfg.warmup_steps, num_training_steps)

    # 9. 训练循环
    print("Start training...")
    global_step = 0
    model.train()
    for epoch in range(cfg.num_epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(cfg.device)
            token_type_ids = batch["token_type_ids"].to(cfg.device)
            attention_mask = batch["attention_mask"].to(cfg.device)
            map_labels = batch["map_labels"].to(cfg.device)
            mfr_targets = batch["mfr_targets"].to(cfg.device)
            mfr_mask = batch["mfr_mask"].to(cfg.device)


            out = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                map_labels=map_labels,
                mfr_targets=mfr_targets,
                mfr_mask=mfr_mask,
            )
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1

            if global_step % 100 == 0:
                print(f"Epoch {epoch+1}, step {global_step}, loss = {loss.item():.4f}, "
                      f"MAP = {out['map_loss'].item():.4f}, "
                      f"MFR = {out['mfr_loss'].item():.4f}, ")

    # 10. 保存模型
    model_path = os.path.join(cfg.output_dir, "as_bert_model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "asn2idx": asn2idx,
        "idx2asn": idx2asn,
        "asnIdx2tokenIdx": asnIdx2tokenIdx,
        "tokenIdx2asnIdx": tokenIdx2asnIdx,
        "config": vars(cfg),
    }, model_path)
    print(f"Model saved to {model_path}")

    # 11. 导出静态 embedding
    print("Exporting static embeddings...")
    model.eval()
    with torch.no_grad():
        emb_static = model.get_as_static_embedding().cpu().numpy()  # (V, d_model)

    # 输出为 TSV: ASN + d 维向量
    static_emb_file = os.path.join(cfg.output_dir, f"as_static_embedding_{cfg.save_name}.txt")
    # with open(static_emb_file, "w") as f:
    #     for idx in range(V):
    #         asn = idx2asn[idx]
    #         vec = emb_static[idx].tolist()
    #         f.write(str(asn) + "\t" + "\t".join(f"{v:.6f}" for v in vec) + "\n")
    # print(f"Static embeddings saved to {static_emb_file}")

    with open(static_emb_file, "w") as f:
        # 写入表头
        dim = len(emb_static[0])
        header = "ASN," + ",".join(f"emb{i+1}" for i in range(dim))
        f.write(header + "\n")

        # 写入每一行 embedding
        for idx in range(V):
            asn = idx2asn[idx]
            vec = emb_static[idx].tolist()
            line = str(asn) + "," + ",".join(f"{v:.8f}" for v in vec)
            f.write(line + "\n")

    print(f"Static embeddings saved to {static_emb_file}")


    # 12. 导出上下文平均 embedding
    print("Building contextual embeddings...")
    dataset = ASPathDataset(paths_idx, asnIdx2tokenIdx, tokenIdx2asnIdx, cfg.max_seq_len, cfg.mask_prob, F_prime, miss_ratio, for_inference = True)
    ctx_dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    emb_ctx = build_contextual_embeddings(model, ctx_dataloader, tokenIdx2asnIdx).cpu().numpy()

    ctx_emb_file = os.path.join(cfg.output_dir, f"as_contextual_embedding_{cfg.save_name}.txt")
    # with open(ctx_emb_file, "w") as f:
    #     for idx in range(V):
    #         asn = idx2asn[idx]
    #         vec = emb_ctx[idx].tolist()
    #         f.write(str(asn) + "\t" + "\t".join(f"{v:.6f}" for v in vec) + "\n")
    # print(f"Contextual embeddings saved to {ctx_emb_file}")

    with open(ctx_emb_file, "w") as f:
        # 写入表头
        dim = len(emb_ctx[0])
        header = "ASN," + ",".join(f"emb{i+1}" for i in range(dim))
        f.write(header + "\n")

        # 写入每一行 embedding
        for idx in range(V):
            asn = idx2asn[idx]
            vec = emb_ctx[idx].tolist()
            line = str(asn) + "," + ",".join(f"{v:.8f}" for v in vec)
            f.write(line + "\n")

    print(f"Contextual embeddings saved to {ctx_emb_file}")


    print("Done.")