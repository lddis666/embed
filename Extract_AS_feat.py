import pandas as pd

def extract_embeddings(input_csv: str, output_csv: str):
    # 需要保留的字段（其中 ASN 为 ID，不参与 emb 编号）
    fields = [
        "AS_rank_numberAsns",
        "AS_rank_numberPrefixes",
        "AS_rank_numberAddresses",
        "AS_rank_total",
        "AS_rank_customer",
        "AS_rank_peer",
        "AS_rank_provider",
        "peeringDB_ix_count",
        "peeringDB_fac_count",
        "AS_hegemony",
        "cti_top",
        "cti_origin"
    ]

    # 读取 CSV, 并将空值设为 -1
    df = pd.read_csv(input_csv).fillna(-1)

    # 若字段不存在（CSV 中为空列），也自动补为 -1
    for col in fields:
        if col not in df.columns:
            df[col] = -1

    # 仅取 ASN + 指定数值字段
    df_selected = df[["ASN"] + fields]

    # 将字段改名为 emb1, emb2 ...
    emb_columns = {field: f"emb{i+1}" for i, field in enumerate(fields)}
    df_selected = df_selected.rename(columns=emb_columns)

    # 保存结果
    df_selected.to_csv(output_csv, index=False)
    print(f"Saved embedding CSV to: {output_csv}")


# 调用示例
extract_embeddings("/mlx_devbox/users/liurundong.991/playground/embed/node_features.csv", "feature_embeddings.csv")
