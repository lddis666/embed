# -*- coding: utf-8 -*-
import logging
import os
import random
import numpy as np
from gensim.models import Word2Vec

random.seed(7)

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s",
    level=logging.INFO
)


class BGP2VEC:
    def __init__(self, model_path, txt_path=None, rewrite=False,
                 embedding_size=128, negative=5, epochs=3, window=2,
                 shuffle=True, word_fq_dict=None):

        self.model_path = model_path
        self.model = None
        self.embedding_size = embedding_size
        self.word_fq_dict = word_fq_dict
        self.routes = None

        if rewrite or not self.import_model():
            if txt_path is None:
                raise ValueError("txt_path 不能为空，用于读取 AS PATH 文件")
            logging.info(f"Start generating BGP2VEC model for {txt_path}")

            # 1. 从 txt 读取 AS PATH
            self.routes = self.load_routes_from_txt(txt_path)

            # 2. 可选：打乱顺序
            if shuffle:
                random.shuffle(self.routes)

            # 3. 训练模型
            self.build_model(embedding_size, window, negative, epochs)

            # 4. 保存模型
            self.export_model()

    @staticmethod
    def load_routes_from_txt(txt_path):
        """
        从 txt 文件读取 AS PATH。
        每一行是一个 AS PATH，AS 之间用空格分隔。
        返回：routes = [ [asn1, asn2, ...], [...], ... ]
        """
        routes = []
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # 按空格分割为 ASN
                asns = line.split()
                routes.append(asns)
        logging.info(f"Loaded {len(routes)} routes from {txt_path}")
        return routes

    def build_model(self, embedding_size, window, negative, epochs):
        # 初始化 Word2Vec 模型
        self.model = Word2Vec(
            vector_size=embedding_size,
            min_count=1,
            window=window,
            sg=1,          # skip-gram
            hs=0,
            negative=negative,
            workers=1,     # 单线程，保证复现性
            seed=7
        )


        # 构建词表
        if self.word_fq_dict:
            self.model.build_vocab_from_freq(self.word_fq_dict)
        else:
            self.model.build_vocab(self.routes, progress_per=100000)

        logging.info(f"Vocabulary size: {len(self.model.wv.key_to_index)}")

        # 训练
        logging.info("Start training model")
        self.model.train(
            self.routes,
            total_examples=len(self.routes),
            epochs=epochs,
            report_delay=30
        )

    def export_model(self):
        self.model.save(self.model_path)
        logging.info(f"Exported model to: {self.model_path}")

    def import_model(self):
        if os.path.exists(self.model_path):
            self.model = Word2Vec.load(self.model_path)
            logging.info(f"Imported existing model from: {self.model_path}")
            return True
        else:
            logging.info(f"Model file not found: {self.model_path}")
            return False

    # 以下是一些辅助方法

    def asn2idx(self, asn):
        return self.model.wv.vocab[asn].index

    def idx2asn(self, idx):
        return self.model.wv.index2word[idx]

    def asn2vec(self, asn):
        return self.model.wv.__getitem__(asn)

    def vec2asn(self, vec):
        return self.model.wv.similar_by_vector(vec, topn=1)[0][0]

    def routes_asn2idx(self, routes, max_len):
        routes_idx = np.zeros([len(routes), max_len], dtype=np.int32)
        for i, route in enumerate(routes):
            for t, asn in enumerate(route):
                if t >= max_len:
                    break
                routes_idx[i, t] = self.asn2idx(asn)
        return routes_idx

    def asns_asn2vec(self, asns):
        asns_vec = np.zeros([len(asns), self.embedding_size])
        for i, asn in enumerate(asns):
            asns_vec[i, :] = self.asn2vec(asn)
        return asns_vec


def export_asn_embeddings_to_txt(model, output_path, embedding_size=128):
    """
    将模型中的每个 ASN 及其向量保存到 txt 文件，CSV 格式：
    第一行：ASN,emb1,emb2,...,emb128
    后面每行：asn,val1,val2,...,val128
    """
    with open(output_path, "w", encoding="utf-8") as f:
        # 写表头
        header = ["ASN"] + [f"emb{i}" for i in range(1, embedding_size + 1)]
        f.write(",".join(header) + "\n")

        # 遍历词表中的每个 ASN
        for asn in model.wv.key_to_index.keys():
            vec = model.wv[asn]
            # 把向量转换为字符串
            vec_str = ",".join(str(float(x)) for x in vec)
            line = f"{asn},{vec_str}\n"
            f.write(line)


if __name__ == "__main__":
    # ===== 1. 配置路径 =====
    # 输入：包含 AS PATH 的 txt 文件（每行一个 AS PATH）
    input_path = "/mlx_devbox/users/liurundong.991/playground/embed/filtered_paths.txt"            # 改成你的 AS PATH 文件路径
    # 中间：Word2Vec 模型文件
    model_path = "./bgp2vec.model"          # 模型保存路径
    # 输出：ASN + 128 维向量的 txt 文件
    output_path = "./asn_embeddings.txt"    # 输出文件路径

    # ===== 2. 训练或加载模型 =====
    bgp2vec = BGP2VEC(
        model_path=model_path,
        txt_path=input_path,
        rewrite=True,          # True: 不管有没有旧模型，都重新训练
        embedding_size=128,    # 你要的 128 维
        negative=5,
        epochs=20,
        window=5,
        shuffle=True,
        word_fq_dict=None
    )

    # ===== 3. 导出 ASN 的向量到 txt =====
    export_asn_embeddings_to_txt(
        model=bgp2vec.model,
        output_path=output_path,
        embedding_size=bgp2vec.embedding_size
    )

    logging.info(f"ASN embeddings exported to: {output_path}")