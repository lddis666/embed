import csv
import random
from typing import List, Dict
import networkx as nx
from gensim.models import Word2Vec


def load_graph(edge_path: str, directed: bool = False) -> nx.Graph:
    """
    从csv文件中读入边，构建图。
    csv格式：
    src_id,dst_id
    1,7843
    1,11537
    ...
    """
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    with open(edge_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = str(row["src_id"])
            dst = str(row["dst_id"])
            G.add_edge(src, dst)

    return G


def random_walk(G: nx.Graph, start_node: str, walk_length: int) -> List[str]:
    """
    普通随机游走（DeepWalk用）
    """
    walk = [start_node]
    while len(walk) < walk_length:
        cur = walk[-1]
        neighbors = list(G.neighbors(cur))
        if len(neighbors) == 0:
            break
        walk.append(random.choice(neighbors))
    return walk


def deepwalk_walks(G: nx.Graph, num_walks: int, walk_length: int) -> List[List[str]]:
    """
    生成DeepWalk的随机游走序列
    """
    nodes = list(G.nodes())
    walks = []
    for _ in range(num_walks):
        random.shuffle(nodes)
        for node in nodes:
            walks.append(random_walk(G, node, walk_length))
    return walks


def node2vec_random_walk(G: nx.Graph, start_node: str, walk_length: int,
                         p: float, q: float) -> List[str]:
    """
    Node2Vec随机游走
    参考论文中的二阶随机游走策略。
    """
    walk = [start_node]
    if walk_length == 1:
        return walk

    neighbors = list(G.neighbors(start_node))
    if len(neighbors) == 0:
        return walk

    # 第一步：普通均匀随机
    first_step = random.choice(neighbors)
    walk.append(first_step)

    # 从第二步开始使用node2vec的转移概率
    while len(walk) < walk_length:
        prev = walk[-2]
        cur = walk[-1]
        cur_neighbors = list(G.neighbors(cur))
        if len(cur_neighbors) == 0:
            break

        probs = []
        for dst in cur_neighbors:
            if dst == prev:
                # 返回上一个节点
                prob = 1.0 / p
            elif G.has_edge(dst, prev) or G.has_edge(prev, dst):
                # 与上一个节点相邻
                prob = 1.0
            else:
                # 远离上一个节点
                prob = 1.0 / q
            probs.append(prob)

        # 归一化
        prob_sum = sum(probs)
        probs = [p_i / prob_sum for p_i in probs]

        # 根据归一化概率选择下一个节点
        r = random.random()
        cum = 0.0
        for dst, prob in zip(cur_neighbors, probs):
            cum += prob
            if r <= cum:
                walk.append(dst)
                break

    return walk


def node2vec_walks(G: nx.Graph, num_walks: int, walk_length: int,
                   p: float, q: float) -> List[List[str]]:
    """
    生成Node2Vec的随机游走序列
    """
    nodes = list(G.nodes())
    walks = []
    for _ in range(num_walks):
        random.shuffle(nodes)
        for node in nodes:
            walks.append(node2vec_random_walk(G, node, walk_length, p, q))
    return walks


def train_skipgram(walks: List[List[str]],
                   embed_dim: int = 128,
                   window_size: int = 5,
                   workers: int = 4,
                   epochs: int = 5) -> Word2Vec:
    """
    使用gensim的Word2Vec进行Skip-Gram训练
    """
    model = Word2Vec(
        sentences=walks,
        vector_size=embed_dim,
        window=window_size,
        min_count=0,
        sg=1,            # 1: skip-gram
        workers=workers,
        epochs=epochs
    )
    return model


def save_embeddings_txt(model: Word2Vec, out_path: str):
    """
    将训练好的embedding保存为txt文件
    格式：
    ASN, embed1, embed2, ...
    1, 0.1, 0.2, ...
    2, ...
    """
    # 所有节点id
    nodes = list(model.wv.key_to_index.keys())
    nodes_sorted = sorted(nodes, key=lambda x: int(x))  # 按数值排序

    dim = model.vector_size
    with open(out_path, "w", encoding="utf-8") as f:
        # 写表头
        header = ["ASN"] + [f"embed{i+1}" for i in range(dim)]
        f.write(",".join(header) + "\n")
        # 写每个节点
        for node in nodes_sorted:
            vec = model.wv[node]
            line = [node] + [f"{v:.6f}" for v in vec]
            f.write(",".join(line) + "\n")


def run_deepwalk(edge_path: str,
                 out_path: str,
                 num_walks: int = 10,
                 walk_length: int = 40,
                 embed_dim: int = 128,
                 window_size: int = 5,
                 workers: int = 4,
                 epochs: int = 5):
    """
    运行DeepWalk流程，从边文件到embedding输出
    """
    print("Loading graph...")
    G = load_graph(edge_path, directed=False)
    print("Generating DeepWalk random walks...")
    walks = deepwalk_walks(G, num_walks=num_walks, walk_length=walk_length)
    print("Training Word2Vec (DeepWalk)...")
    model = train_skipgram(
        walks,
        embed_dim=embed_dim,
        window_size=window_size,
        workers=workers,
        epochs=epochs
    )
    print(f"Saving embeddings to {out_path} ...")
    save_embeddings_txt(model, out_path)
    print("Done (DeepWalk).")


def run_node2vec(edge_path: str,
                 out_path: str,
                 num_walks: int = 10,
                 walk_length: int = 40,
                 p: float = 1.0,
                 q: float = 1.0,
                 embed_dim: int = 128,
                 window_size: int = 5,
                 workers: int = 4,
                 epochs: int = 5):
    """
    运行Node2Vec流程，从边文件到embedding输出
    """
    print("Loading graph...")
    G = load_graph(edge_path, directed=False)
    print("Generating Node2Vec random walks...")
    walks = node2vec_walks(G,
                           num_walks=num_walks,
                           walk_length=walk_length,
                           p=p,
                           q=q)
    print("Training Word2Vec (Node2Vec)...")
    model = train_skipgram(
        walks,
        embed_dim=embed_dim,
        window_size=window_size,
        workers=workers,
        epochs=epochs
    )
    print(f"Saving embeddings to {out_path} ...")
    save_embeddings_txt(model, out_path)
    print("Done (Node2Vec).")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train AS embeddings with DeepWalk or Node2Vec.")
    parser.add_argument("--edge_path", type=str, default="/mlx_devbox/users/liurundong.991/playground/embed/dataset/raw_edges.csv",
                        help="Path to edge csv file.")
    parser.add_argument("--method", type=str, choices=["deepwalk", "node2vec"],
                        default="node2vec", help="Method: deepwalk or node2vec.")
    parser.add_argument("--output", type=str, default="node2vec_embeddings.txt",
                        help="Output txt file for embeddings.")
    parser.add_argument("--num_walks", type=int, default=10,
                        help="Number of walks per node.")
    parser.add_argument("--walk_length", type=int, default=40,
                        help="Length of each walk.")
    parser.add_argument("--dim", type=int, default=128,
                        help="Embedding dimension.")
    parser.add_argument("--window", type=int, default=5,
                        help="Word2Vec window size.")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of worker threads.")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Training epochs.")
    parser.add_argument("--p", type=float, default=1.0,
                        help="Node2Vec parameter p (return).")
    parser.add_argument("--q", type=float, default=1.0,
                        help="Node2Vec parameter q (in-out).")

    args = parser.parse_args()

    if args.method == "deepwalk":
        run_deepwalk(edge_path=args.edge_path,
                     out_path=args.output,
                     num_walks=args.num_walks,
                     walk_length=args.walk_length,
                     embed_dim=args.dim,
                     window_size=args.window,
                     workers=args.workers,
                     epochs=args.epochs)
    else:
        run_node2vec(edge_path=args.edge_path,
                     out_path=args.output,
                     num_walks=args.num_walks,
                     walk_length=args.walk_length,
                     p=args.p,
                     q=args.q,
                     embed_dim=args.dim,
                     window_size=args.window,
                     workers=args.workers,
                     epochs=args.epochs)