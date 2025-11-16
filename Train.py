#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AS-BERT : BERT-like self-supervised pre-training for Autonomous System embeddings
author : your_name
"""

import json, math, random, argparse, os, itertools, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

# -------------------------------------------------
# Utils
# -------------------------------------------------
def set_seed(seed=2023):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------------------------------
# 1. Data  -------------------------------------------------
class ASPairDataset(Dataset):
    """
    负责：
      1. 读一条 AS‐PATH 行 → 拆成两个 segment (50% 正样本 / 50% 换 B 段负样本)
      2. 在线随机 MASK 15% token，用于 MLM+MFR
    返回：
      dict{input_ids, token_type_ids, attn_mask,
           mlm_labels, feat_labels, is_next}
    """
    def __init__(self,
                 path_file,
                 vocab_size,
                 max_len=128,
                 mask_prob=0.15,
                 cls_id=0, sep_id=1, mask_id=2):
        self.lines = open(path_file).read().strip().splitlines()
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.vocab_size = vocab_size
        self.special_ids = {'[CLS]': cls_id, '[SEP]': sep_id, '[MASK]': mask_id}

    def __len__(self): return len(self.lines)

    # ---- 辅助：把路径切成两半 ----
    def _split_path(self, tokens):
        if len(tokens) < 2: return tokens, []
        cut = random.randint(1, len(tokens) - 1)
        return tokens[:cut], tokens[cut:]

    @staticmethod
    def _truncate(toks_a, toks_b, max_len_inner):
        """ 截断到 max_len_inner """
        while len(toks_a) + len(toks_b) > max_len_inner:
            if len(toks_a) > len(toks_b):
                toks_a.pop()
            else:
                toks_b.pop()

    # ---- MLM & MFR mask 选择 ----
    def _mask_tokens(self, input_ids):
        """
        返回：
          masked_input_ids, mlm_labels(where -100 means ignore)
        """
        mlm_labels = [-100] * len(input_ids)
        for i in range(len(input_ids)):
            # 跳过 special token
            if input_ids[i] in self.special_ids.values(): continue
            if random.random() < self.mask_prob:
                # 保存 label
                mlm_labels[i] = input_ids[i]
                p = random.random()
                if p < .8:           # 80% -> [MASK]
                    input_ids[i] = self.special_ids['[MASK]']
                elif p < .9:         # 10% -> random token
                    input_ids[i] = random.randrange(3, self.vocab_size)
                else:                # 10% -> keep
                    pass
        return input_ids, mlm_labels

    def __getitem__(self, idx):
        line_a = self.lines[idx].strip().split()
        toks_a = [int(x) for x in line_a]

        is_next = 1  # 1=True(续接), 0=False
        toks_b = []

        if random.random() < 0.5:
            # 50 % 正样本
            toks_a, toks_b = self._split_path(toks_a)
        else:
            # 50 % 负样本
            is_next = 0
            toks_a, _tmp = self._split_path(toks_a)
            rand_idx = random.randrange(len(self.lines))
            toks_b = [int(x) for x in self.lines[rand_idx].split()]
            # 取后半段
            _, toks_b = self._split_path(toks_b)

        # truncate inner len = max_len-3([CLS],[SEP],[SEP])
        self._truncate(toks_a, toks_b, self.max_len - 3)

        # 拼装句子
        cls, sep = self.special_ids['[CLS]'], self.special_ids['[SEP]']
        input_ids = [cls] + toks_a + [sep] + toks_b + [sep]
        token_type_ids = [0]*(len(toks_a)+2) + [1]*(len(toks_b)+1)
        attn_mask = [1]*len(input_ids)

        # padding
        pad_len = self.max_len - len(input_ids)
        input_ids.extend([0]*pad_len)
        token_type_ids.extend([0]*pad_len)
        attn_mask.extend([0]*pad_len)

        # 动态 MLM
        input_ids_masked, mlm_labels = self._mask_tokens(input_ids.copy())

        return {
            'input_ids': torch.tensor(input_ids_masked ,dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids,dtype=torch.long),
            'attention_mask': torch.tensor(attn_mask     ,dtype=torch.long),
            'mlm_labels': torch.tensor(mlm_labels        ,dtype=torch.long),
            'is_next': torch.tensor(is_next              ,dtype=torch.long)
        }

# -------------------------------------------------
# 2. Model  -------------------------------------------------
class ASBert(nn.Module):
    def __init__(self, cfg, features, miss_mask):
        super().__init__()
        self.cfg = cfg
        V, K = features.shape
        self.hidden_size = cfg.hidden
        # ----- 2K 连续+缺失 -----
        # 预处理：z-score & 0 fill 已在加载阶段完成
        self.feat_table = nn.Embedding.from_pretrained(
            torch.tensor(np.concatenate([features, miss_mask], axis=1),
                         dtype=torch.float32),
            freeze=True)                      # (V, 2K)
        # miss ratio  (V,1)
        miss_ratio = miss_mask.mean(1, keepdims=True)
        self.miss_ratio = nn.Embedding.from_pretrained(
            torch.tensor(miss_ratio, dtype=torch.float32),
            freeze=True)

        # ----- Embeddings -----
        self.id_embed   = nn.Embedding(V, cfg.hidden, padding_idx=0)
        self.pos_embed  = nn.Embedding(cfg.max_len, cfg.hidden)
        self.seg_embed  = nn.Embedding(2, cfg.hidden)

        self.feat_proj = nn.Sequential(
            nn.Linear(2*K, cfg.hidden),
            nn.GELU(),
            nn.Linear(cfg.hidden, cfg.hidden)
        )
        # gate parameters
        self.gate_w = nn.Parameter(torch.zeros(1, cfg.hidden))
        self.gate_b = nn.Parameter(torch.zeros(1, cfg.hidden))

        self.emb_layernorm = nn.LayerNorm(cfg.hidden)
        self.dropout       = nn.Dropout(cfg.dropout)

        # ----- Encoder -----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden,
            nhead=cfg.nhead,
            dim_feedforward=cfg.ffn,
            dropout=cfg.dropout,
            activation='gelu',
            batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer,
                                             num_layers=cfg.layers)

        # ----- Heads -----
        self.mlm_head   = nn.Linear(cfg.hidden, V)
        self.mfr_head   = nn.Linear(cfg.hidden, 2*K)
        self.nsp_head   = nn.Linear(cfg.hidden, 2)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.id_embed.weight, 0, 0.02)
        nn.init.normal_(self.pos_embed.weight, 0, 0.02)
        nn.init.normal_(self.seg_embed.weight, 0, 0.02)
        for m in [self.mlm_head, self.mfr_head, self.nsp_head]:
            nn.init.normal_(m.weight, 0, 0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)

    # ------------ Static Embedding （一并给出接口） ------------
    def get_static_embeddings(self):
        """
        返回 tensor (V , hidden)
        """
        id_emb = self.id_embed.weight            # (V,d)
        feat_vec = self.feat_proj(self.feat_table.weight)  # (V,d)
        gate = torch.sigmoid(self.gate_w * self.miss_ratio.weight + self.gate_b) # (V,d)
        emb = self.emb_layernorm(id_emb + gate * feat_vec)
        return emb.detach()

    # ------------ forward ------------
    def forward(self, input_ids, token_type_ids, attention_mask,
                mlm_labels=None,   # (B,L)
                is_next=None):     # (B,)
        B, L = input_ids.shape
        device = input_ids.device
        # ----- Embedding lookup -----
        id_emb   = self.id_embed(input_ids)                       # (B,L,d)
        pos_ids  = torch.arange(L, device=device).unsqueeze(0).expand(B,L)
        pos_emb  = self.pos_embed(pos_ids)
        seg_emb  = self.seg_embed(token_type_ids)

        feat_vec = self.feat_proj(self.feat_table(input_ids))      # (B,L,d)
        miss_r   = self.miss_ratio(input_ids)                      # (B,L,1)
        gate     = torch.sigmoid(self.gate_w*miss_r + self.gate_b) # (B,L,d)
        emb = self.emb_layernorm(id_emb + gate*feat_vec + pos_emb + seg_emb)
        emb = self.dropout(emb)

        # ----- Encoder -----
        attn_key = attention_mask == 0   # bool pad mask
        hidden = self.encoder(emb, src_key_padding_mask=attn_key)  # (B,L,d)

        # outputs
        seq_output = hidden
        pooled_output = hidden[:,0]  # CLS

        logits_mlm = self.mlm_head(seq_output)            # (B,L,V)
        logits_mfr = self.mfr_head(seq_output)            # (B,L,2K)
        logits_nsp = self.nsp_head(pooled_output)         # (B,2)

        total_loss = None
        if mlm_labels is not None and is_next is not None:
            # -------- Loss --------
            loss_fct_ce = nn.CrossEntropyLoss()
            mlm_loss = loss_fct_ce(
                logits_mlm.view(-1, logits_mlm.size(-1)),
                mlm_labels.view(-1))

            # MFR 只在被 mask 的位置计算
            mask_pos = mlm_labels.ne(-100).unsqueeze(-1).expand_as(logits_mfr)
            feat_labels = self.feat_table(input_ids)      # (B,L,2K)
            mfr_loss = F.smooth_l1_loss(
                logits_mfr[mask_pos],
                feat_labels[mask_pos],
                reduction='mean') if mask_pos.any() else torch.tensor(0., device=device)

            nsp_loss = loss_fct_ce(logits_nsp, is_next)

            total_loss = mlm_loss + self.cfg.lam_mfr*mfr_loss + self.cfg.lam_nsp*nsp_loss

        return total_loss, (logits_mlm, logits_mfr, logits_nsp)

# -------------------------------------------------
# 3. Train  -------------------------------------------------
def train(cfg):
    device = get_device()
    print(f"Device: {device}")
    # ------------ 3.1 读取词表 + 特征 ------------
    vocab = json.load(open(cfg.vocab))
    vocab_size = len(vocab)
    features = np.load(cfg.feature)      # (V,K)
    K = features.shape[1]
    # z-score  (用全部数据统计量，亦可分 train/val)
    mu, sigma = np.nanmean(features, axis=0), np.nanstd(features, axis=0)+1e-6
    features = (features - mu) / sigma
    miss_mask = np.isnan(features).astype(np.float32)
    features = np.nan_to_num(features, nan=0.0).astype(np.float32)

    # ------------ 3.2 DataLoader ------------
    ds = ASPairDataset(cfg.path, vocab_size,
                       max_len=cfg.max_len)
    dl = DataLoader(ds, batch_size=cfg.bs, shuffle=True,
                    num_workers=4, pin_memory=torch.cuda.is_available())

    # ------------ 3.3 Model ------------
    model = ASBert(cfg, features, miss_mask).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    # ------------ 3.4 训练循环 ------------
    model.train()
    global_step = 0
    for epoch in range(1, cfg.epochs+1):
        for batch in dl:
            batch = {k: v.to(device) for k,v in batch.items()}
            loss,_ = model(**batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); optimizer.zero_grad()
            global_step += 1
            if global_step % 100 == 0:
                print(f"epoch {epoch} step {global_step} loss {loss.item():.4f}")
        # 每个 epoch 保存一次
        save_ckpt(model, optimizer, cfg, epoch)
    print("Finished training!")

    # ------------ 3.5 导出 Embedding ------------
    emb_static = model.get_static_embeddings().cpu().numpy()
    np.save(cfg.output_dir / 'embedding_static.npy', emb_static)
    print("Static embedding saved.")

    # ------------ 3.6 Contextual Mean Pooling ------------
    ctx_sum = torch.zeros((vocab_size, cfg.hidden), dtype=torch.float32, device=device)
    ctx_cnt = torch.zeros((vocab_size, 1),            dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        for batch in dl:
            input_ids = batch['input_ids'].to(device)
            att_mask  = batch['attention_mask'].to(device)
            token_type= batch['token_type_ids'].to(device)
            _, (hidden, _,_) = model(input_ids, token_type, att_mask)
            # hidden : (B,L,d)
            for b in range(hidden.size(0)):
                ids = input_ids[b]       # (L,)
                valid = att_mask[b].bool()
                ids = ids[valid]; hid = hidden[b,valid]
                ctx_sum.index_add_(0, ids, hid)
                ctx_cnt.index_add_(0, ids, torch.ones_like(ids, dtype=torch.float32).unsqueeze(1))
    ctx_mean = (ctx_sum/ctx_cnt.clamp(min=1e-6)).cpu().numpy()
    np.save(cfg.output_dir / 'embedding_ctxmean.npy', ctx_mean)
    print("Contextual mean embedding saved.")

def save_ckpt(model, optim, cfg, epoch):
    out = cfg.output_dir / f"ckpt_epoch{epoch}.pt"
    torch.save({'model':model.state_dict(),
                'optim':optim.state_dict(),
                'epoch':epoch}, out)
    print(f"[saved] {out}")

# -------------------------------------------------
# 4. Argparse / Config  -------------------------------------------------
def get_cfg():
    p = argparse.ArgumentParser()
    # files
    p.add_argument('--path',    type=Path, required=True, help='as_path.txt')
    p.add_argument('--feature', type=Path, required=True, help='as_features.npy')
    p.add_argument('--vocab',   type=Path, required=True, help='vocab.json')
    p.add_argument('--output_dir', type=Path, default=Path('./outputs'))
    # hyper
    p.add_argument('--max_len', type=int, default=128)
    p.add_argument('--hidden',  type=int, default=256)
    p.add_argument('--nhead',   type=int, default=4)
    p.add_argument('--layers',  type=int, default=4)
    p.add_argument('--ffn',     type=int, default=1024)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--lr',      type=float, default=3e-4)
    p.add_argument('--bs',      type=int, default=64)
    p.add_argument('--epochs',  type=int, default=5)
    # loss weights
    p.add_argument('--lam_mfr', type=float, default=1.0)
    p.add_argument('--lam_nsp', type=float, default=1.0)
    args = p.parse_args()
    args.output_dir.mkdir(exist_ok=True, parents=True)
    return args

# -------------------------------------------------
if __name__ == '__main__':
    set_seed()
    cfg = get_cfg()
    train(cfg)




# {"0": 0, "1": 1, "6453": 2, "3356": 3, ...}

# 其中 0 1 2 为 [PAD] [CLS] [SEP]（可自行调整，只要与脚本中 special id 对应即可）
# • as_features.npy
# numpy float32, shape=(|V| , K) ，缺失用 np.nan。

# python as_bert.py \
#    --path data/as_path.txt \
#    --feature data/as_features.npy \
#    --vocab data/vocab.json \
#    --epochs 10 --bs 128 --hidden 256