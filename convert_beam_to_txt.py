import pickle

input_file = "/mlx_devbox/users/liurundong.991/playground/routing-anomaly-detection-master/BEAM_engine/models/20240801.as-rel2.1000.10.128/node.emb"
output_file = "/mlx_devbox/users/liurundong.991/playground/embed/dataset/beam.txt"

# 读取 pickle
with open(input_file, "rb") as f:
    data = pickle.load(f)   # { ASN : numpy_array }

# 任意取一个 embedding 的维度来写 header
dim = len(next(iter(data.values())))

# 写入 TXT
with open(output_file, "w") as f:
    # 写表头：ASN,emb1,emb2,...,embN
    header = "ASN," + ",".join(f"emb{i+1}" for i in range(dim))
    f.write(header + "\n")

    # 写数据
    for asn, emb in data.items():
        emb_str = ",".join(str(x) for x in emb)
        f.write(f"{asn},{emb_str}\n")

print("已生成：", output_file)
