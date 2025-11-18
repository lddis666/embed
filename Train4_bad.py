# train.py
# o4mini
import os
import random
import argparse
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# -------------------------
#  Hyper-parameters
# -------------------------
DEFAULT_MAX_LEN    = 128
DEFAULT_D_MODEL    = 128
DEFAULT_N_LAYERS   = 4
DEFAULT_N_HEADS    = 8
DEFAULT_MLP_HID    = 512
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS     = 10
MASK_PROB          = 0.15
LAMBDA_MAP         = 1.0
LAMBDA_MFR         = 1.0
LAMBDA_NSP         = 0.5

# -------------------------
#  Utility: read feature file
# -------------------------
def load_features(feat_path):
    """
    feat.txt 格式: ASN, emb1, emb2, ..., embK
    返回:
        asn2idx: dict[asn] -> idx
        feat_mat_raw: np.array |V| x K   （原始特征；缺失用 -1）
    """
    lines = open(feat_path).read().strip().splitlines()
    header = lines[0].split(',')
    K = len(header) - 1
    V = len(lines) - 1
    asn2idx = {}
    feat_mat_raw = np.zeros((V, K), dtype=float)
    for i, line in enumerate(lines[1:]):
        arr = line.strip().split(',')
        asn = arr[0].strip()
        asn2idx[asn] = i
        vals = [float(x) for x in arr[1:]]
        feat_mat_raw[i] = np.array(vals, dtype=float)
    return asn2idx, feat_mat_raw  # shape (V,K)

# -------------------------
#  Dataset
# -------------------------
class ASPathDataset(Dataset):
    def __init__(self, path_txt, asn2idx, feat_mat_raw, max_len):
        """
        path_txt: 每行一个 AS PATH (space-separated)
        asn2idx: feature 中出现的 ASN->idx
        feat_mat_raw: 原始连续特征矩阵, shape (V,K)
        """
        self.max_len = max_len
        self.asn2idx = asn2idx
        self.feat_mat_raw = feat_mat_raw  # 用于后续 F', M
        self.K = feat_mat_raw.shape[1]

        # 1) 读取所有路径，映射到 idx
        raw_paths = []
        for L in open(path_txt):
            seq = [w.strip() for w in L.strip().split() if w.strip()!='']
            # 过滤那些在 feat 中不出现的 ASN
            seq = [asn for asn in seq if asn in asn2idx]
            if len(seq) >= 2:
                raw_paths.append(seq)
        self.paths = raw_paths
        self.N = len(self.paths)

        # 2) 特征预处理：F' = concat(F_processed, M)
        #    F_processed: 将 -1 -> 0, M-mask 下标记录
        Fp = []
        for i in range(self.feat_mat_raw.shape[0]):
            row = self.feat_mat_raw[i].copy()
            mask = (row < 0).astype(float)  # 缺失部分
            row[row < 0] = 0.0
            Fp.append(np.concatenate([row, mask], axis=0))
        self.feat_prime = np.stack(Fp, axis=0).astype(np.float32)  # (V,2K)
        # 3) miss_ratio
        miss_ratio = (self.feat_mat_raw < 0).sum(axis=1) / float(self.K)
        self.miss_ratio = miss_ratio.astype(np.float32)  # (V,)

        # 4) 构造 vocab：只有那些在 feat 中出现的 ASN
        self.idx2asn = {i:asn for asn,i in asn2idx.items()}
        self.V = len(self.idx2asn)
        # 特殊符号
        self.pad_id  = self.V
        self.cls_id  = self.V + 1
        self.sep_id  = self.V + 2
        self.mask_id = self.V + 3
        self.vocab_size = self.V + 4

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        """
        1) 随机 NSP 采样生成 A|B, seg_ids, is_next_label
        2) 返回原始序列(AS idx), seg_ids, is_next_label
        注意：MLM/MFR 的随机 mask 在 collate 中做
        """
        seq = self.paths[idx]
        # positive / negative NSP
        if random.random() < 0.5:
            # 正样本：从 seq 中 cut
            if len(seq) < 3:
                # 太短则直接 A=prefix, B empty
                A = seq[:-1]
                B = seq[-1:]
            else:
                k = random.randint(1, len(seq)-1)
                A = seq[:k]
                B = seq[k:]
            is_next = 1
        else:
            # 负样本：A 来自 seq, B 来自另一条随机 path
            k = random.randint(1, len(seq)-1)
            A = seq[:k]
            # pick other
            j = random.randint(0, self.N-1)
            while j == idx:
                j = random.randint(0, self.N-1)
            seq2 = self.paths[j]
            if len(seq2) < 1:
                B = seq2
            else:
                k2 = random.randint(1, len(seq2))
                B = seq2[:k2]
            is_next = 0

        # 拼接 [CLS] A [SEP] B [SEP]
        ids = [self.cls_id] + [ self.asn2idx[asn] for asn in A ] \
              + [self.sep_id] \
              + [ self.asn2idx[asn] for asn in B ] \
              + [self.sep_id]
        seg_ids = [0]*(1+len(A)) + [1]*(1+len(B))  # CLS & A 段 0, B 段 1
        return {
            'ids': ids,
            'seg': seg_ids,
            'is_next': is_next
        }

def collate_fn(batch, dataset:ASPathDataset):
    """
    batch: list of dict from __getitem__
    输出一个大 dict，包含：
     ids, seg_ids, attn_mask,
     feat_ids [B, L, 2K], miss_ratio_ids [B,L],
     mlm_input_ids, mlm_labels, mlm_mask
     mfr_labels [B,L,2K],
     next_labels [B]
    """
    pad_id = dataset.pad_id
    mask_id = dataset.mask_id
    B = len(batch)
    L = min(dataset.max_len, max(len(x['ids']) for x in batch))
    # init
    ids = torch.full((B, L), pad_id, dtype=torch.long)
    seg = torch.zeros((B, L), dtype=torch.long)
    attn_mask = torch.zeros((B, L), dtype=torch.bool)
    feat_ids = torch.zeros((B, L, dataset.feat_prime.shape[1]), dtype=torch.float32)
    miss_ratio_ids = torch.zeros((B, L), dtype=torch.float32)
    next_labels = torch.zeros(B, dtype=torch.long)

    for i, item in enumerate(batch):
        cur_ids = item['ids'][:L]
        cur_len = len(cur_ids)
        ids[i, :cur_len] = torch.tensor(cur_ids, dtype=torch.long)
        seg[i, :cur_len] = torch.tensor(item['seg'][:L], dtype=torch.long)
        attn_mask[i, :cur_len] = True
        next_labels[i] = item['is_next']
        # feat & miss_ratio
        # idx 0..V-1 maps to feat_prime row
        feat_ids[i, :cur_len, :] = torch.from_numpy(dataset.feat_prime[cur_ids])
        miss_ratio_ids[i, :cur_len] = torch.from_numpy(dataset.miss_ratio[cur_ids])

    # 生成 MLM & MFR mask
    mlm_input = ids.clone()
    mlm_labels = torch.full((B, L), -100, dtype=torch.long)
    mfr_labels = torch.zeros((B, L, dataset.feat_prime.shape[1]), dtype=torch.float32)
    mlm_mask = torch.zeros((B, L), dtype=torch.bool)

    for i in range(B):
        for j in range(L):
            token = ids[i,j].item()
            # 只对非 PAD/CLS/SEP 做 mask
            if attn_mask[i,j] and token < dataset.V and random.random() < MASK_PROB:
                mlm_mask[i,j] = True
                mlm_labels[i,j] = token
                mfr_labels[i,j] = feat_ids[i,j]
                prob = random.random()
                if prob < 0.8:
                    mlm_input[i,j] = mask_id
                elif prob < 0.9:
                    # 随机替换
                    mlm_input[i,j] = random.randint(0, dataset.V-1)
                else:
                    mlm_input[i,j] = token
    return {
        'input_ids':     mlm_input,
        'seg_ids':       seg,
        'attn_mask':     attn_mask,
        'feat_ids':      feat_ids,
        'miss_ratio':    miss_ratio_ids,
        'mlm_labels':    mlm_labels,
        'mlm_mask':      mlm_mask,
        'mfr_labels':    mfr_labels,
        'next_labels':   next_labels
    }

# -------------------------
#  Model
# -------------------------
class ASBert(nn.Module):
    def __init__(self,
                 vocab_size, feat_dim, d_model,
                 n_heads, n_layers, mlp_hid, max_len):
        super().__init__()
        self.d_model = d_model
        self.emb_id = nn.Embedding(vocab_size, d_model)
        self.emb_pos = nn.Embedding(max_len, d_model)
        self.emb_seg = nn.Embedding(2, d_model)

        # feature MLP
        self.mlp_feat = nn.Sequential(
            nn.Linear(feat_dim, mlp_hid),
            nn.GELU(),
            nn.Linear(mlp_hid, d_model)
        )
        # gate: 1-d input -> 1
        self.gate = nn.Linear(1, 1)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=mlp_hid,
            dropout=0.1,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # heads
        self.head_mlm = nn.Linear(d_model, vocab_size)
        self.head_mfr = nn.Linear(d_model, feat_dim)
        self.head_nsp = nn.Linear(d_model, 2)

        # static embedding layernorm
        self.ln_static = nn.LayerNorm(d_model)

    def forward(self, input_ids, seg_ids, attn_mask, feat_ids, miss_ratio):
        """
        input_ids: [B,L]
        seg_ids:    [B,L]
        attn_mask:  [B,L] bool
        feat_ids:   [B,L,feat_dim]
        miss_ratio: [B,L]
        """
        B, L = input_ids.shape
        device = input_ids.device
        # token embedding
        h_id = self.emb_id(input_ids)               # [B,L,d]
        # feature embed
        h_f = self.mlp_feat(feat_ids)                # [B,L,d]
        g  = torch.sigmoid(self.gate(miss_ratio.unsqueeze(-1)))  # [B,L,1]
        h_f = h_f * g
        # pos & seg
        pos_idx = torch.arange(L, device=device).unsqueeze(0).expand(B,-1)
        h_pos = self.emb_pos(pos_idx)
        h_seg = self.emb_seg(seg_ids)
        # sum
        h = h_id + h_f + h_pos + h_seg            # [B,L,d]
        # transformer: needs [L,B,d]
        key_mask = ~attn_mask                     # padding positions = True
        h = h.permute(1,0,2)  # [L,B,d]
        h = self.transformer(h, src_key_padding_mask=key_mask)
        h = h.permute(1,0,2)  # [B,L,d]

        # heads
        mlm_logits = self.head_mlm(h)             # [B,L,vocab]
        mfr_pred   = self.head_mfr(h)             # [B,L,feat_dim]
        cls_emb    = h[:,0]                       # [B,d]
        nsp_logits = self.head_nsp(cls_emb)       # [B,2]
        return mlm_logits, mfr_pred, nsp_logits

    def export_static_embeddings(self, dataset:ASPathDataset, out_path):
        """
        导出静态 embedding: emb = LayerNorm( id_emb + gate*feat_emb )
        输出 txt: ASN \t d-dim
        """
        self.eval()
        with torch.no_grad():
            V = dataset.V
            # idx 0..V-1
            id_idxs = torch.arange(V, device=next(self.parameters()).device)
            x_id  = self.emb_id(id_idxs)                      # [V,d]
            feat = torch.from_numpy(dataset.feat_prime).to(x_id.device)  # [V,2K]
            h_f  = self.mlp_feat(feat)                        # [V,d]
            miss = torch.from_numpy(dataset.miss_ratio).unsqueeze(1).to(x_id.device)  # [V,1]
            g    = torch.sigmoid(self.gate(miss))             # [V,1]
            emb  = self.ln_static(x_id + h_f * g)             # [V,d]
            emb = emb.cpu().numpy()
        # 写出
        with open(out_path, 'w') as fw:
            for idx in range(V):
                asn = dataset.idx2asn[idx]
                vec = ' '.join([f'{x:.6f}' for x in emb[idx]])
                fw.write(f'{asn}\t{vec}\n')

# -------------------------
#  Train & Eval
# -------------------------
def train(args):
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 数据 & Dataset
    asn2idx, feat_mat_raw = load_features(args.feat_txt)
    dataset = ASPathDataset(
        path_txt=args.path_txt,
        asn2idx=asn2idx,
        feat_mat_raw=feat_mat_raw,
        max_len=args.max_len
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, dataset),
        num_workers=2,
        pin_memory=True
    )
    # model
    model = ASBert(
        vocab_size=dataset.vocab_size,
        feat_dim=dataset.feat_prime.shape[1],
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        mlp_hid=args.mlp_hid,
        max_len=args.max_len
    ).to(device)
    # optimizer & scheduler
    optimizer = AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs*len(dataloader), eta_min=1e-6)
    # loss fn
    loss_mlm = nn.CrossEntropyLoss(ignore_index=-100)
    loss_nsp = nn.CrossEntropyLoss()
    loss_mfr = nn.SmoothL1Loss()

    # train loop
    model.train()
    for epoch in range(1, args.epochs+1):
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}")
        for i, batch in pbar:
            # to device
            for k,v in batch.items():
                batch[k] = v.to(device)
            mlm_logits, mfr_pred, nsp_logits = model(
                batch['input_ids'],
                batch['seg_ids'],
                batch['attn_mask'],
                batch['feat_ids'],
                batch['miss_ratio']
            )
            # MLM loss
            B,L = batch['input_ids'].shape
            mlm_loss = loss_mlm(
                mlm_logits.view(B*L, -1),
                batch['mlm_labels'].view(B*L)
            )
            # MFR loss: 只对被 mask 的位置
            mfr_mask = batch['mlm_mask']  # [B,L]
            if mfr_mask.any():
                # flatten选中
                pred_sel = mfr_pred[mfr_mask]
                lab_sel  = batch['mfr_labels'][mfr_mask]
                mfr_loss = loss_mfr(pred_sel, lab_sel)
            else:
                mfr_loss = torch.tensor(0., device=device)
            # NSP
            nsp_loss = loss_nsp(nsp_logits, batch['next_labels'])
            # total
            loss = LAMBDA_MAP*mlm_loss + LAMBDA_MFR*mfr_loss + LAMBDA_NSP*nsp_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            pbar.set_postfix({
                'L':loss.item(),
                'MAP':mlm_loss.item(),
                'MFR':mfr_loss.item(),
                'NSP':nsp_loss.item()
            })

        # 每 epoch 导出一次静态 embedding
        out_static = os.path.join(args.save_dir, f'static_emb_epoch{epoch}.txt')
        model.export_static_embeddings(dataset, out_static)
        print(f'-- saved static embeddings to {out_static}')

    # 最后再导出一次
    out_static = os.path.join(args.save_dir, 'static_emb_final.txt')
    model.export_static_embeddings(dataset, out_static)
    print(f'-- saved final static embeddings to {out_static}')

# -------------------------
#  main & arg parse
# -------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_txt', type=str, required=True,
                        help='每行一个 AS-PATH 的 txt 文件')
    parser.add_argument('--feat_txt', type=str, required=True,
                        help='AS 特征 txt 文件，第一列 ASN，后面  K 列数值')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='输出模型与 embedding 保存目录')
    parser.add_argument('--max_len',  type=int, default=DEFAULT_MAX_LEN)
    parser.add_argument('--d_model',  type=int, default=DEFAULT_D_MODEL)
    parser.add_argument('--n_layers', type=int, default=DEFAULT_N_LAYERS)
    parser.add_argument('--n_heads',  type=int, default=DEFAULT_N_HEADS)
    parser.add_argument('--mlp_hid',  type=int, default=DEFAULT_MLP_HID)
    parser.add_argument('--batch_size',type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--epochs',   type=int, default=DEFAULT_EPOCHS)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    train(args)