#!/usr/bin/env python3  
# -*- coding: utf-8 -*-

import argparse
from collections import Counter

# 判断是否是私有 ASN（64512–65534 为常见 16-bit private ASN 范围）
def is_private_asn(asn: int) -> bool:
    return 64512 <= asn <= 65534

# 解析一条 AS PATH 字符串为整数 list
def parse_as_path(line: str):
    """
    1. 删除大括号 / AS-SET（这里简单地跳过含有 “{”“}” 或逗号 “,” 的行）
    2. 按空白切分
    3. 转为 int
    """
    line = line.strip()
    if not line:
        return None
    # 如果包含 AS-SET 的语法，跳过此路径
    if '{' in line or '}' in line or ',' in line:
        return None
    parts = line.split()
    asns = []
    for p in parts:
        try:
            asn = int(p)
            asns.append(asn)
        except ValueError:
            # 如果不是纯数字，比如有前缀或注释，跳过该条路径
            return None
    return asns

# 压缩连续重复 AS（例如 loop 或重复跳）
def compress_consecutive(as_path):
    """把相邻重复的 ASN 压缩为一个"""
    if not as_path:
        return []
    comp = [as_path[0]]
    for a in as_path[1:]:
        if a != comp[-1]:
            comp.append(a)
    return comp

def load_and_count(input_file):
    """第一遍：读取所有 path，解析并计数每个 ASN 出现频率"""
    as_counter = Counter()
    parsed_paths = []  # list of (原始 line, parsed asns list)
    with open(input_file, 'r') as f:
        for lineno, line in enumerate(f, start=1):
            parsed = parse_as_path(line)
            if parsed is None:
                continue
            parsed = compress_consecutive(parsed)
            parsed_paths.append((line.rstrip('\n'), parsed))
            for a in parsed:
                as_counter[a] += 1
    return parsed_paths, as_counter

def filter_paths(parsed_paths, as_counter,
                 min_as_frequency: int,
                 max_path_len: int,
                 drop_private_asn: bool):
    """第二遍：根据频率、私有 ASN、最大长度来过滤路径"""
    rare_as = {a for a, cnt in as_counter.items() if cnt < min_as_frequency}
    output_paths = set()  # 用 set 去重 (tuple 形式)
    for original_line, asns in parsed_paths:
        # 过滤私有 ASN
        if drop_private_asn and any(is_private_asn(a) for a in asns):
            continue
        # 过滤包含低频 ASN 的路径
        if any(a in rare_as for a in asns):
            continue
        # 过滤过长路径
        if len(asns) > max_path_len:
            continue
        # 压缩后再检查长度（可选）
        comp = compress_consecutive(asns)
        if len(comp) > max_path_len:
            continue
        # 最终去重：转成 tuple
        tup = tuple(comp)
        output_paths.add(tup)
    return output_paths

def write_output(output_paths, output_file):
    """把过滤后的 path 写入文件（每行 space 分隔）"""
    with open(output_file, 'w') as f:
        for asns in sorted(output_paths):
            line = ' '.join(str(a) for a in asns)
            f.write(line + "\n")

def main():
    parser = argparse.ArgumentParser(description="Clean AS PATH 文件")
    parser.add_argument("--input", "-i", required=True, help="输入 AS PATH txt 文件，每行一个 path")
    parser.add_argument("--output", "-o", required=True, help="清洗后输出文件")
    parser.add_argument("--min_as_freq", "-m", type=int, default=10,
                        help="AS 最少出现次数 (低于此频率的 AS 将被视为 rare 并剔除包含它的路径)")
    parser.add_argument("--max_path_len", "-l", type=int, default=10,
                        help="允许的最大 AS 路径长度（跳数）")
    parser.add_argument("--drop_private", "-p", action="store_true",
                        help="是否剔除包含私有 ASN 的路径 (如 64512–65534)")

    args = parser.parse_args()

    print("第一遍加载和计数 …")
    parsed_paths, as_counter = load_and_count(args.input)
    print(f"总共读入 {len(parsed_paths)} 条（去除语法不合规的可能更少）")
    print(f"共识别 {len(as_counter)} 个不同 ASN")

    print("过滤 …")
    output_paths = filter_paths(parsed_paths, as_counter,
                                min_as_frequency=args.min_as_freq,
                                max_path_len=args.max_path_len,
                                drop_private_asn=args.drop_private)
    print(f"过滤后剩余 {len(output_paths)} 条唯一路径")

    print("写入输出 …")
    write_output(output_paths, args.output)
    print("完成。")

if __name__ == "__main__":
    main()


# python clean_as_paths.py -i paths.txt -o filtered_paths.txt -m 20 -l 8 -p