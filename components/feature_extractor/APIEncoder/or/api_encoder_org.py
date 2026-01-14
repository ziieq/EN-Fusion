from sys import exception
import torch
from m1_end.fe.APIEncoder.lib.bert_encoder.bert_encoder import BertEncoder
from .orthogonalizer_v2_768 import BatchAPIOrthogonalizer
import numpy as np


class APIEncoder:
    def __init__(self, api2intro_path='./api2intro.txt', bert_pkl_path='api2intro',
                 is_raw=False, orthogonal_strength=0.2, min_semantic_preservation=0.8, is_orthogonalize=True):
        with open(api2intro_path, 'r', encoding='utf-8') as f:
            self.api2intro = eval(f.read())
        self.bert = BertEncoder(bert_pkl_path)
        self.is_raw = is_raw

        self.orthogonalizer = BatchAPIOrthogonalizer(similarity_threshold= 0.7,
                 orthogonal_strength= orthogonal_strength,
                 min_semantic_preservation=min_semantic_preservation)
        self.is_orthogonalize = is_orthogonalize
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

        if self.is_orthogonalize:
            new_enhanced_embeddings = self.orthogonalizer.orthogonalize(np.array(embedding_list), api_name_list)
        else:
            new_enhanced_embeddings = torch.stack(embedding_list)
        for i, (api, embedding) in enumerate(zip(api_name_list, new_enhanced_embeddings)):
            self.api2embedding[api] = embedding

        # 计算相似度统计
        orig_similarities = []
        enhanced_similarities = []
        n_apis = len(api_name_list)
        for i in range(n_apis):
            for j in range(i + 1, n_apis):
                # 原始相似度
                orig_sim = self.orthogonalizer.compute_similarity_corrected(
                    embedding_list[i], embedding_list[j]
                )
                orig_similarities.append(orig_sim)

                # 增强后相似度
                enhanced_sim = self.orthogonalizer.compute_similarity_corrected(
                    new_enhanced_embeddings[i], new_enhanced_embeddings[j]
                )
                enhanced_similarities.append(enhanced_sim)

        print(f"原始嵌入平均相似度: {np.mean(orig_similarities):.4f}")
        print(f"正交化后平均相似度: {np.mean(enhanced_similarities):.4f}")
        print(
            f"相似度降低比例: {(np.mean(orig_similarities) - np.mean(enhanced_similarities)) / np.mean(orig_similarities) * 100:.1f}%")

    def encode(self, api_name):
        if api_name not in self.api2embedding:
            raise exception(rf'{api_name} not in api2embedding!')
        return self.api2embedding[api_name]
