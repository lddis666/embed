def merge_embedding_files(file1, file2, out_file, has_header=True, sep=','):
    """
    将两个 embedding 文件按 ASN 对齐并拼接，只保留 ASN 交集，输出到 out_file。
    
    参数：
        file1, file2 : str
            输入 embedding 文件路径
        out_file     : str
            输出文件路径
        has_header   : bool
            若为 True，则认为每个输入文件首行是表头并跳过
        sep          : str
            列分隔符，默认为逗号 ','
            
    文件格式假设：
        每行：ASN,emb1,emb2,emb3,...
        示例：1,-1.0,0.5,0.3
    """
    def load_embeddings(path):
        emb_dict = {}
        with open(path, 'r', encoding='utf-8') as f:
            first_line = True
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # 处理表头
                if first_line and has_header:
                    first_line = False
                    continue
                first_line = False

                parts = line.split(sep)
                asn = parts[0].strip()
                try:
                    vec = [float(x) for x in parts[1:]]
                except ValueError:
                    # 遇到异常行就跳过
                    continue
                emb_dict[asn] = vec
        return emb_dict

    emb1 = load_embeddings(file1)
    emb2 = load_embeddings(file2)

    # ASN 交集
    common_asn = sorted(set(emb1.keys()) & set(emb2.keys()),
                        key=lambda x: int(x) if x.isdigit() else x)

    with open(out_file, 'w', encoding='utf-8') as out:
        # 写表头（简单写 ASN + 维度索引，如需更复杂可以自行改）
        if common_asn:
            d1 = len(emb1[common_asn[0]])
            d2 = len(emb2[common_asn[0]])
            cols = ['ASN'] + [f'f1_{i}' for i in range(d1)] + [f'f2_{i}' for i in range(d2)]
            out.write(sep.join(cols) + '\n')

        # 写数据
        for asn in common_asn:
            v1 = emb1[asn]
            v2 = emb2[asn]
            merged = v1 + v2
            row = [asn] + [str(x) for x in merged]
            out.write(sep.join(row) + '\n')


merge_embedding_files(
    file1="./output/as_static_embedding_1203-map-mfr-no-missing-indicator.txt",
    file2="./output/as_contextual_embedding_1203-map-mfr-no-missing-indicator.txt",
    out_file="./output/as_contextual_embedding_1203-merge.txt",
    has_header=True,  # 如果原文件没有表头就改为 False
    sep=','           # 如果是空格分隔可以改为 ' '
)