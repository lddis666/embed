import bz2

def remove_duplicates(route):
    """
    移除相邻重复的ASN，比如 [1, 1, 2, 2, 3] 变成 [1, 2, 3]
    """
    if not route:
        return []
    non_dup_route = [route[0]]
    for asn in route[1:]:
        if asn != non_dup_route[-1]:
            non_dup_route.append(asn)
    return non_dup_route

def extract_as_paths_from_bz2(input_bz2_path, output_txt_path):
    with bz2.open(input_bz2_path, "rt") as infile, open(output_txt_path, "w") as outfile:
        for line in infile:
            cols = line.strip().split()
            # 判断是否为有效的路由行（以*开头且字段数足够）
            if len(cols) > 6 and cols[0] == '*':
                # 提取AS PATH（通常为第7列开始到倒数第2列）
                as_path = cols[6:-1]
                # 去除相邻重复ASN
                as_path = remove_duplicates(as_path)
                if as_path:  # 非空路径才写入
                    outfile.write(" ".join(as_path) + "\n")

# 用法示例
extract_as_paths_from_bz2("/Users/ldd/Desktop/embed/dataset/oix-full-snapshot-2025-01-01-0200.bz2", "AS_PATH.txt")