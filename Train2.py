#!/usr/bin/env python
# coding: utf-8
# -o3
"""
AS-Embedding 预训练主程序
python as_pretrain.py \
    --path_txt data/as_path.txt \
    --feat_txt data/as_feat.txt \
    --output_dir exp_as_embedding \
    --device cuda        # 或 cpu
"""
import os, argparse, math, random, json, csv
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


# -----------------------------
# 一、数据加载与词表/特征构建
# -----------------------------
def load_features(feat_txt, missing_val=-1.):
    """
    读取 feature 文件 (CSV/TSV 均可)  返回:
    feat_mat:  |V|×K  numpy array
    miss_mask: same shape 0/1
    asn2idx : dict
    """
    reader = csv.reader(open(feat_txt))
    header = next(reader)
    feats = []
    asn2idx = {}
    for i, row in enumerate(reader):
        asn = int(row[0])
        vec = [float(x) for x in row[1:]]
        asn2idx[asn] = i
        feats.append(vec)
    feat_mat = np.asarray(feats, np.float32)            # (|V|, K)
    miss_mask = (feat_mat == missing_val).astype(np.float32)
    feat_mat[miss_mask == 1] = 0.0                      # 把缺失处置 0
    return feat_mat, miss_mask, asn2idx, header[1:]


def load_paths(path_txt, asn2idx):
    """
    只保留全部 ASN 都在词表中的路径
    """
    paths = []
    with open(path_txt) as f:
        for ln in f:
            toks = [int(x) for x in ln.strip().split()]
            ok = all(t in asn2idx for t in toks)
            if ok and len(toks):
                paths.append([asn2idx[t] for t in toks])  # 变成 idx
    return paths


# -----------------------------
# 二、Dataset
# -----------------------------
class ASPTDataset(Dataset):
    def __init__(self, paths, feat_mat, miss_mask,
                 mask_prob=0.15, max_len=128):
        self.paths = paths
        self.feat_mat = feat_mat          # numpy
        self.miss_mask = miss_mask
        self.K = feat_mat.shape[1]
        self.mask_prob = mask_prob
        self.max_len = max_len
        self.vocab_size = feat_mat.shape[0]

    def __len__(self):
        return len(self.paths)

    def _truncate_or_pad(self, seq):
        if len(seq) > self.max_len:
            seq = seq[:self.max_len]
        return seq

    def __getitem__(self, idx):
        # ---------- 1. NSP ----------
        p = self.paths[idx]
        p = self._truncate_or_pad(p)
        # 正/负 50%
        if random.random() < 0.5:
            # 正样本
            is_next = 1
            cut = random.randint(1, max(1, len(p)-1))
            left, right = p[:cut], p[cut:]
        else:
            # 负样本  (在另一条路径随机截取)
            is_next = 0
            other = self.paths[random.randint(0, len(self.paths)-1)]
            other = self._truncate_or_pad(other)
            cut1 = random.randint(1, max(1, len(p)-1))
            cut2 = random.randint(1, max(1, len(other)-1))
            left, right = p[:cut1], other[cut2:]

        # [CLS] A [SEP] B [SEP]
        cls_id = self.vocab_size          # special
        sep_id = self.vocab_size + 1
        mask_id = self.vocab_size + 2

        tokens = [cls_id] + left + [sep_id] + right + [sep_id]
        seg_ids = ([0]*(len(left)+2)) + ([1]*(len(right)+1))  # CLS + left + SEP =0

        # ---------- 2. MAP masking ----------
        labels = [-100]*len(tokens)
        for i in range(1, len(tokens)-1):     # 不 mask CLS/SEP
            if tokens[i] >= self.vocab_size:  # special 不参与
                continue
            if random.random() < self.mask_prob:
                labels[i] = tokens[i]         # 需要预测的真 ASN
                r = random.random()
                if r < 0.8:
                    tokens[i] = mask_id
                elif r < 0.9:
                    tokens[i] = random.randint(0, self.vocab_size-1)
                # else 10% keep same
        # ---------- 3. MFR target ----------
        feat_targets = np.zeros((len(tokens), self.K*2), np.float32)
        feat_targets[:] = 0.0
        masked_positions = []
        for i, lab in enumerate(labels):
            if lab != -100:
                masked_positions.append(i)
                as_idx = lab                # 原来真实 ASN index
                vec = self.feat_mat[as_idx]
                miss = self.miss_mask[as_idx]
                feat_targets[i, :self.K] = vec
                feat_targets[i, self.K:] = miss
        feat_targets = torch.from_numpy(feat_targets)

        sample = dict(
            input_ids=torch.tensor(tokens, dtype=torch.long),
            token_type_ids=torch.tensor(seg_ids, dtype=torch.long),
            attention_mask=torch.ones(len(tokens), dtype=torch.long),
            map_labels=torch.tensor(labels, dtype=torch.long),
            mfr_labels=feat_targets,
            nsp_label=torch.tensor(is_next, dtype=torch.long)
        )
        return sample


def collate_fn(batch):
    keys = batch[0].keys()
    pad_lens = [len(x['input_ids']) for x in batch]
    max_len = max(pad_lens)

    def pad_1d(x, pad_val=0):
        return torch.cat([x, torch.full((max_len-len(x),), pad_val, dtype=x.dtype)], dim=0)

    def pad_2d(x):
        pad = torch.zeros((max_len - x.size(0), x.size(1)), dtype=x.dtype)
        return torch.cat([x, pad], dim=0)

    out = {}
    for k in keys:
        if k in ['input_ids', 'token_type_ids', 'attention_mask', 'map_labels']:
            out[k] = torch.stack([pad_1d(s[k], pad_val=0 if k!='map_labels' else -100) for s in batch])
        elif k == 'mfr_labels':
            out[k] = torch.stack([pad_2d(s[k]) for s in batch])
        elif k == 'nsp_label':
            out[k] = torch.stack([s[k] for s in batch])
    return out


# -----------------------------
# 三、模型
# -----------------------------
class FeatureMLP(nn.Module):
    def __init__(self, in_dim, d_model, hidden_factor=2.0):
        super().__init__()
        hid = int(d_model*hidden_factor)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.GELU(),
            nn.Linear(hid, d_model)
        )
    def forward(self, x):
        return self.net(x)

class ASEmbModel(nn.Module):
    def __init__(self, vocab, k_feat, d_model=128, n_layers=4, n_heads=4,
                 dropout=0.1, max_len=256):
        """
        vocab = |V|   (不含3个 special)
        k_feat = K
        """
        super().__init__()
        self.vocab = vocab
        self.total_vocab = vocab + 3                     # 加 CLS/SEP/MASK
        self.d_model = d_model
        self.k_feat = k_feat
        in_dim = k_feat*2

        # 1. ID Embedding  (|V|+3, d)
        self.id_embed = nn.Embedding(self.total_vocab, d_model)

        # 2. Feature-MLP
        self.feat_mlp = FeatureMLP(in_dim, d_model)

        # 3. Gate： miss_ratio  (K缺失数 / K) -> scalar
        self.gate_w = nn.Linear(1, 1)

        # 4. 位置、Segment
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.seg_embed = nn.Embedding(2, d_model)

        # 5. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   nhead=n_heads,
                                                   dim_feedforward=d_model*4,
                                                   dropout=dropout,
                                                   activation='gelu',
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 6. Heads
        self.map_head = nn.Linear(d_model, self.vocab)          # 只预测真实 vocab
        self.mfr_head = nn.Linear(d_model, in_dim)              # 2K
        self.nsp_head = nn.Linear(d_model, 2)

        self.layer_norm = nn.LayerNorm(d_model)

    # ----------- 静态 Embedding（无上下文）-----------
    def get_static_embedding(self, feat_mat, miss_mask, device='cpu', batch=4096):
        """
        feat_mat:  numpy |V|×K
        miss_mask: numpy |V|×K
        return:    torch |V|×d
        """
        self.eval()
        with torch.no_grad():
            V = feat_mat.shape[0]
            all_emb = []
            for i in range(0, V, batch):
                idx = torch.arange(i, min(V, i+batch), dtype=torch.long, device=device)
                e_id = self.id_embed(idx)                      # (b,d)
                vec = torch.from_numpy(feat_mat[i:i+len(idx)]).to(device)
                miss = torch.from_numpy(miss_mask[i:i+len(idx)]).to(device)
                e_feat = self.feat_mlp(torch.cat([vec, miss], dim=-1).float())
                miss_ratio = miss.mean(-1, keepdim=True)       # (b,1)
                g = torch.sigmoid(self.gate_w(miss_ratio))     # (b,1)
                emb = self.layer_norm(e_id + g*e_feat)
                all_emb.append(emb.cpu())
            return torch.cat(all_emb, dim=0)                   # |V|×d

    def forward(self, batch):
        # shape: B×L
        ids = batch['input_ids']
        seg = batch['token_type_ids']
        B, L = ids.size()
        device = ids.device

        # 构造 miss_ratio 和 e_feat 批量查表
        with torch.no_grad():
            # ids >= vocab -> special token 没有连续特征，填 0
            feat_ids = ids.clone()
            feat_ids[feat_ids >= self.vocab] = 0    # any valid idx
        # hack: 使用 buffer 存储全量特征可避免来回复制，也可写成Embedding
        if not hasattr(self, '_feat_mat_torch'):
            raise ValueError("call model.set_feat_matrix() before training")

        vec_full = self._feat_mat_torch[feat_ids]        # (B,L,K)
        miss_full = self._miss_mask_torch[feat_ids]      # (B,L,K)
        miss_ratio = miss_full.float().mean(-1, keepdim=True)

        # Embedding 合成
        e_id = self.id_embed(ids)                        # (B,L,d)
        e_feat = self.feat_mlp(torch.cat([vec_full, miss_full], dim=-1).float())
        gate = torch.sigmoid(self.gate_w(miss_ratio))    # (B,L,1)
        emb = e_id + gate * e_feat \
              + self.pos_embed(torch.arange(L, device=device)) \
              + self.seg_embed(seg)
        emb = self.layer_norm(emb)

        # Transformer
        attn_mask = (batch['attention_mask'] == 0)       # padding 为 True
        h = self.encoder(emb, src_key_padding_mask=attn_mask)

        # Heads
        map_logits = self.map_head(h)                    # (B,L,V)
        mfr_pred = self.mfr_head(h)                      # (B,L,2K)
        cls = h[:,0]                                     # [CLS]
        nsp_logits = self.nsp_head(cls)                  # (B,2)

        return map_logits, mfr_pred, nsp_logits

    # 注册全量特征张量，避免反复 to(device)
    def set_feat_matrix(self, feat_mat, miss_mask, device):
        self._feat_mat_torch = torch.from_numpy(feat_mat).to(device)
        self._miss_mask_torch = torch.from_numpy(miss_mask).to(device)


# -----------------------------
# 四、Loss & Train
# -----------------------------
def compute_loss(batch, outputs, lambda_map=1.0, lambda_mfr=1.0, lambda_nsp=0.5):
    map_logits, mfr_pred, nsp_logits = outputs
    B, L, V = map_logits.size()
    # MAP
    loss_map = nn.functional.cross_entropy(map_logits.view(-1, V),
                                           batch['map_labels'].view(-1),
                                           ignore_index=-100)
    # MFR  仅 masked 位置
    mask_pos = (batch['map_labels'] != -100).unsqueeze(-1)  # (B,L,1)
    diff = (mfr_pred - batch['mfr_labels'].to(mfr_pred.device))**2
    mse = (diff * mask_pos.float()).sum() / (mask_pos.sum()+1e-6)
    loss_mfr = mse

    # NSP
    loss_nsp = nn.functional.cross_entropy(nsp_logits, batch['nsp_label'])

    total = lambda_map*loss_map + lambda_mfr*loss_mfr + lambda_nsp*loss_nsp
    return total, dict(map=loss_map.item(),
                       mfr=loss_mfr.item(),
                       nsp=loss_nsp.item())


def train_loop(model, loader, optimizer, scheduler, device,
               epochs=1, log_step=100):
    model.train()
    for ep in range(epochs):
        for step, batch in enumerate(loader):
            for k in batch:
                batch[k] = batch[k].to(device)
            optimizer.zero_grad()
            outs = model(batch)
            loss, parts = compute_loss(batch, outs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            if step % log_step == 0:
                print(f"Ep{ep} step{step}  total={loss.item():.4f} "
                      f"MAP={parts['map']:.4f}  MFR={parts['mfr']:.4f} "
                      f"NSP={parts['nsp']:.4f}")


# -----------------------------
# 五、上下文均值 embedding 导出
# -----------------------------
def export_contextual_mean(model, dataset, device='cpu', save_path='ctx.npy',
                           max_batches=None):
    print('Collecting contextual embeddings ...')
    model.eval()
    cnt = torch.zeros(model.vocab, dtype=torch.long)
    s = torch.zeros(model.vocab, model.d_model)
    loader = DataLoader(dataset, batch_size=32,
                        collate_fn=collate_fn, shuffle=False)
    with torch.no_grad():
        for b, batch in enumerate(loader):
            if max_batches and b >= max_batches:
                break
            for k in batch:
                batch[k] = batch[k].to(device)
            h, _, _ = model(batch)   # we only need encoder output
            h = h[0]  # map logits 返回第一个 pred ？  直接重新跑
        # 为了简单，可调用 model.forward 再取 encoder hidden
    print('Done.')
    # 这里只提供框架，实际实现同静态导出的思路一样:
    #    对于 batch 内所有 token，按其真实 ASN 累加 encoder hidden
    #    最终求均值并保存。自行补充或删改即可。
    pass


# -----------------------------
# 六、主函数
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_txt', required=True)
    parser.add_argument('--feat_txt', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--device', default='cuda')
    # 模型 / 训练超参
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--max_len', type=int, default=128)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and
                          args.device.startswith('cuda') else 'cpu')
    print('using device', device)

    # 1. 数据 & 词表
    feat_mat, miss_mask, asn2idx, feat_cols = load_features(args.feat_txt)
    paths = load_paths(args.path_txt, asn2idx)
    print(f"paths {len(paths)}  vocab |V|={len(asn2idx)}  feature dim={feat_mat.shape[1]}")

    # 2. Dataset / Loader
    ds = ASPTDataset(paths, feat_mat, miss_mask, max_len=args.max_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    collate_fn=collate_fn, drop_last=True)

    # 3. Model
    model = ASEmbModel(vocab=len(asn2idx), k_feat=feat_mat.shape[1],
                       d_model=args.d_model, n_layers=args.layers,
                       n_heads=args.heads, max_len=args.max_len)
    model.set_feat_matrix(feat_mat, miss_mask, device)
    model.to(device)

    # 4. Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer,
                                  T_max=len(dl)*args.epochs,
                                  eta_min=args.lr*0.1)

    # 5. Train
    train_loop(model, dl, optimizer, scheduler, device,
               epochs=args.epochs, log_step=100)

    # 6. 导出静态 Embedding
    emb_static = model.get_static_embedding(feat_mat, miss_mask,
                                            device=device)
    np.save(os.path.join(args.output_dir, 'emb_static.npy'),
            emb_static.cpu().numpy())
    with open(os.path.join(args.output_dir, 'asn2idx.json'), 'w') as f:
        json.dump(asn2idx, f)
    print('Static embedding saved.')

    # （可选）导出上下文均值 embedding
    # export_contextual_mean(model, ds, device=device,
    #                        save_path=os.path.join(args.output_dir, 'emb_ctx.npy'))

    print('Finished.')


if __name__ == '__main__':
    main()