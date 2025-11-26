import bz2
import csv

# 输入 / 输出文件路径（请根据实际路径修改）
INPUT_FILE = "20250901.as-rel2.txt.bz2"
OUTPUT_FILE = "./dataset/as_relations_onehot.txt"

# 关系类型说明（按 CAIDA as-rel 规范）：
#   0  : Peer-to-Peer (P2P)
#  -1  : Provider-to-Customer (P2C);  反向为 Customer-to-Provider (C2P)

def main():
    # 打开输出 txt 文件
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        # 写表头
        out_f.write("ASN1,ASN2,P2P,P2C,C2P\n")

        # 读取 bz2 压缩的 as-rel2 文件
        with bz2.open(INPUT_FILE, "rt", encoding="utf-8") as in_f:
            reader = csv.reader(in_f, delimiter='|')
            for row in reader:
                # 跳过注释行（以 # 开头）
                if not row or row[0].startswith('#'):
                    continue

                # row 典型结构: [AS1, AS2, rel_type, (optional extra...)]
                try:
                    as1 = row[0].strip()
                    as2 = row[1].strip()
                    rel = int(row[2].strip())
                except (IndexError, ValueError):
                    # 如果行格式不对或不是整数关系，直接跳过
                    continue

                # 只保留 P2P (0) 和 P2C (-1)
                if rel not in (0, -1):
                    continue

                # 情况 1：P2P (0)
                if rel == 0:
                    # P2P 是对等关系，两边都 P2P
                    # AS1 -> AS2
                    out_f.write(f"{as1},{as2},1,0,0\n")
                    # AS2 -> AS1
                    out_f.write(f"{as2},{as1},1,0,0\n")

                # 情况 2：P2C (-1)
                elif rel == -1:
                    # 按常用约定：AS1 是 provider，AS2 是 customer
                    # AS1 -> AS2 : Provider-to-Customer (P2C)
                    out_f.write(f"{as1},{as2},0,1,0\n")
                    # AS2 -> AS1 : Customer-to-Provider (C2P)
                    out_f.write(f"{as2},{as1},0,0,1\n")

    print(f"已生成文件: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()