import csv

input_file = "/mlx_devbox/users/liurundong.991/playground/embed/output/as_static_embedding.tsv"      # 你的输入文件
output_file = "/mlx_devbox/users/liurundong.991/playground/embed/output/as_static_embedding.txt"  # 输出文件

with open(input_file, "r") as fin, open(output_file, "w", newline="") as fout:
    writer = csv.writer(fout)

    # 读取第一行，确定 embedding 维度
    first_line = fin.readline().strip().split("\t")
    dim = len(first_line) - 1  # 第一个是 ASN，其余是 embedding 维度

    # 写表头
    header = ["ASN"] + [f"emb{i}" for i in range(1, dim + 1)]
    writer.writerow(header)

    # 写第一行数据
    writer.writerow(first_line)

    # 写剩余数据
    for line in fin:
        parts = line.strip().split("\t")
        writer.writerow(parts)

print("转换完成，已保存到:", output_file)
