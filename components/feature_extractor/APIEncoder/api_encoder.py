from .lib.bert_encoder.bert_encoder import BertEncoder
import numpy as np
from .whitening import Whitening
import torch
from .deepseek.deepseek_requester import DeepSeek

def efficient_mean_cosine_similarity(vectors):
    """
    高效计算平均余弦相似度，避免存储完整的相似度矩阵

    Args:
        vectors: 形状(n, d)的数组
        return_matrix: 是否返回相似度矩阵

    Returns:
        平均余弦相似度
    """
    if isinstance(vectors, torch.Tensor):
        vectors_np = vectors.detach().cpu().numpy()
    else:
        vectors_np = vectors

    n, d = vectors_np.shape

    # 1. 归一化
    norms = np.linalg.norm(vectors_np, axis=1)
    norms = np.maximum(norms, 1e-8)
    normalized = vectors_np / norms[:, np.newaxis]

    # 2. 高效计算总相似度
    total_similarity = 0

    # 分批处理，避免大矩阵
    batch_size = min(1000, n)  # 调整批次大小

    for i in range(0, n, batch_size):
        end_i = min(i + batch_size, n)
        batch = normalized[i:end_i]

        # 计算当前批次与其他所有向量的相似度
        similarities = batch @ normalized.T

        # 减去对角线（如果当前批次包含其他批次的元素）
        for j in range(len(batch)):
            idx = i + j
            similarities[j, idx] = 0  # 置零对角线元素

        total_similarity += np.sum(similarities)

    mean_similarity = total_similarity / (n * n - n)
    return mean_similarity

class APIEncoder:
    def __init__(self, api2intro_path='./api2intro.txt', bert_pkl_path='api2intro',
                 is_raw=False, dimention=128):
        self.api2intro_path = api2intro_path
        with open(api2intro_path, 'r', encoding='utf-8') as f:
            self.api2intro = eval(f.read())
        self.bert = BertEncoder(bert_pkl_path)
        self.is_raw = is_raw
        self.whitener = Whitening(n_components=dimention)
        self.ds = DeepSeek()
        self.api2embedding = {}
        self.init()


    def init(self):
        api_name_list, embedding_list = [], []
        for api, intro in self.api2intro.items():
            api_name_list.append(api)
            if self.is_raw:
                em = self.bert.encode_str(api)[0]
            else:
                em = self.bert.encode_str(self.api2intro[api])[0]
            embedding_list.append(em)

        embedding_list = np.array(embedding_list)
        self.whitener.fit(embedding_list)
        new_enhanced_embeddings = self.whitener.transform(embedding_list)

        print('余弦相似度：')
        print(efficient_mean_cosine_similarity(new_enhanced_embeddings))

        for i, (api, embedding) in enumerate(zip(api_name_list, new_enhanced_embeddings)):
            self.api2embedding[api] = embedding


    def encode(self, api_name):
        if api_name not in self.api2embedding:
            # 遇到没见过的api
            intro = self.ds.request(api_name)
            self.api2intro[api_name] = intro
            with open(self.api2intro_path, 'w', encoding='utf-8') as f:
                f.write(str(self.api2intro))

            if self.is_raw:
                em = self.bert.encode_str(api_name)[0]
            else:
                em = self.bert.encode_str(self.api2intro[api_name])[0]
            embedding = self.whitener.transform([em])[0]
            self.api2embedding[api_name] = embedding
            return embedding
        else:
            return self.api2embedding[api_name]

    def compute_similarity_corrected(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        修正的相似度计算：直接计算整个嵌入的余弦相似度

        参数:
            emb1, emb2: 完整嵌入向量（包含语义和正交部分）

        返回:
            余弦相似度
        """
        # 直接计算整个向量的余弦相似度
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

