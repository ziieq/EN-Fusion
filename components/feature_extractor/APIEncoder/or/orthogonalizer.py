import torch
import tqdm
import sys

import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple
import faiss  # 用于高效相似度搜索


class APIEmbeddingOrthogonalizer:
    def __init__(self,
                 base_embed_dim: int = 128,
                 orthogonal_strength: float = 0.3,
                 semantic_preserve_weight: float = 0.7,
                 ortho_threshold: float = 0.5):
        """
        参数:
            base_embed_dim: 基础嵌入维度
            orthogonal_strength: 正交化强度系数(0-1)
            semantic_preserve_weight: 语义保留权重
            ortho_threshold: 正交化阈值
        """
        self.base_dim = base_embed_dim
        # 扩展维度以容纳正交分量
        self.total_dim = int(base_embed_dim * (1 + orthogonal_strength))
        self.ortho_dim = self.total_dim - base_embed_dim

        self.ortho_strength = orthogonal_strength
        self.semantic_weight = semantic_preserve_weight
        self.threshold = ortho_threshold

        # 存储旧API嵌入（固定）
        self.old_api_embeddings = {}
        self.old_api_names = []

        # FAISS索引用于快速相似度搜索
        self.index = None
        self.is_index_trained = False

    def initialize_old_apis(self, api_names: List[str],
                            embeddings: np.ndarray):
        """初始化旧API嵌入（固定不变）"""
        for name, emb in zip(api_names, embeddings):
            # 如果提供的嵌入维度不足，填充零
            if emb.shape[0] < self.total_dim:
                padded_emb = np.zeros(self.total_dim)
                padded_emb[:emb.shape[0]] = emb
            else:
                padded_emb = emb[:self.total_dim]

            self.old_api_embeddings[name] = padded_emb
            self.old_api_names.append(name)

        # 构建FAISS索引
        self._build_faiss_index()

    def _build_faiss_index(self):
        """构建FAISS索引用于快速相似度计算"""
        if not self.old_api_embeddings:
            return

        embeddings_array = np.array(list(self.old_api_embeddings.values()))
        self.index = faiss.IndexFlatIP(self.total_dim)  # 内积相似度
        self.index.add(embeddings_array.astype(np.float32))
        self.is_index_trained = True

    def compute_orthogonal_component(self,
                                     base_embedding: np.ndarray,
                                     reference_embeddings: np.ndarray) -> np.ndarray:
        """
        计算正交分量（施密特正交化）

        参数:
            base_embedding: 基础语义嵌入
            reference_embeddings: 参考嵌入（已有API）

        返回:
            正交分量
        """
        # 归一化基础嵌入
        u = base_embedding / (np.linalg.norm(base_embedding) + 1e-8)

        # 对每个参考嵌入进行正交化
        for ref in reference_embeddings:
            ref_norm = ref / (np.linalg.norm(ref) + 1e-8)
            # 投影
            projection = np.dot(u, ref_norm) * ref_norm
            # 减去投影分量
            u = u - projection

        # 重新归一化
        u = u / (np.linalg.norm(u) + 1e-8)
        return u * self.ortho_strength

    def learn_new_api_embedding(self,
                                api_name: str,
                                base_embedding: np.ndarray,
                                context_embeddings: List[np.ndarray] = None) -> np.ndarray:
        """
        学习新API的嵌入

        参数:
            api_name: API名称
            base_embedding: 预训练的基础语义嵌入
            context_embeddings: 上下文相关API的嵌入

        返回:
            增强后的嵌入
        """
        # 步骤1: 确保基础嵌入维度正确
        if base_embedding.shape[0] < self.base_dim:
            # 填充到base_dim维度
            padded_base = np.zeros(self.base_dim)
            padded_base[:base_embedding.shape[0]] = base_embedding
            base_embedding = padded_base
        elif base_embedding.shape[0] > self.base_dim:
            base_embedding = base_embedding[:self.base_dim]

        # 步骤2: 获取最近的k个旧API嵌入作为参考
        if self.is_index_trained:
            # 使用FAISS搜索相似API
            query = np.zeros(self.total_dim)
            query[:self.base_dim] = base_embedding

            k = min(10, len(self.old_api_names))
            distances, indices = self.index.search(
                query.reshape(1, -1).astype(np.float32), k)

            # 获取相似度超过阈值的参考嵌入
            reference_embs = []
            for dist, idx in zip(distances[0], indices[0]):
                if dist > self.threshold:  # 内积相似度大于阈值
                    ref_name = self.old_api_names[idx]
                    reference_embs.append(self.old_api_embeddings[ref_name])
        else:
            reference_embs = []

        # 步骤3: 添加上下文相关API（如果有）
        if context_embeddings:
            for ctx_emb in context_embeddings:
                if ctx_emb.shape[0] == self.total_dim:
                    reference_embs.append(ctx_emb)

        # 步骤4: 计算正交分量
        if reference_embs:
            reference_array = np.array(reference_embs)
            # 只对参考嵌入的基础维度部分进行正交化
            reference_base = reference_array[:, :self.base_dim]
            ortho_component = self.compute_orthogonal_component(
                base_embedding, reference_base)
        else:
            # 如果没有参考，随机生成正交分量
            ortho_component = np.random.randn(self.ortho_dim)
            ortho_component = ortho_component / np.linalg.norm(ortho_component)
            ortho_component = ortho_component * self.ortho_strength

        # 步骤5: 组合基础嵌入和正交分量
        final_embedding = np.zeros(self.total_dim)
        final_embedding[:self.base_dim] = base_embedding * self.semantic_weight

        # 将正交分量放置在后半部分
        ortho_start = self.base_dim
        ortho_end = ortho_start + min(self.ortho_dim, ortho_component.shape[0])
        final_embedding[ortho_start:ortho_end] = ortho_component[:ortho_end - ortho_start]

        # 归一化最终嵌入
        final_embedding = final_embedding / (np.linalg.norm(final_embedding) + 1e-8)

        return final_embedding

    def batch_learn_new_apis(self,
                             api_names: List[str],
                             base_embeddings: np.ndarray,
                             similarity_matrix: np.ndarray = None) -> np.ndarray:
        """
        批量学习新API嵌入（考虑新API之间的正交性）

        参数:
            api_names: API名称列表
            base_embeddings: 基础嵌入矩阵 [n_api, base_dim]
            similarity_matrix: API之间的相似度矩阵

        返回:
            增强后的嵌入矩阵
        """
        n_apis = len(api_names)
        enhanced_embeddings = np.zeros((n_apis, self.total_dim))

        # 第一步：独立学习每个API的嵌入（只考虑与旧API的正交性）
        for i, (name, base_emb) in enumerate(zip(api_names, base_embeddings)):
            enhanced_embeddings[i] = self.learn_new_api_embedding(name, base_emb)

        # 第二步：调整新API之间的正交性
        if similarity_matrix is not None and n_apis > 1:
            enhanced_embeddings = self._adjust_inter_api_orthogonality(
                enhanced_embeddings, similarity_matrix)

        return enhanced_embeddings

    def _adjust_inter_api_orthogonality(self,
                                        embeddings: np.ndarray,
                                        similarity_matrix: np.ndarray) -> np.ndarray:
        """
        调整新API之间的正交性
        使用迭代正交化过程
        """
        n = embeddings.shape[0]
        adjusted = embeddings.copy()

        # 正交化迭代次数
        n_iterations = 3

        for _ in range(n_iterations):
            for i in range(n):
                # 获取当前API的基础部分
                base_i = adjusted[i, :self.base_dim]

                # 找出相似度高的其他API
                similar_indices = np.where(similarity_matrix[i] > self.threshold)[0]
                similar_indices = similar_indices[similar_indices != i]

                if len(similar_indices) > 0:
                    # 获取相似API的基础部分
                    similar_bases = adjusted[similar_indices, :self.base_dim]

                    # 计算正交调整
                    for j, sim_base in enumerate(similar_bases):
                        sim = similarity_matrix[i, similar_indices[j]]
                        if sim > self.threshold:
                            # 计算正交分量
                            sim_norm = sim_base / (np.linalg.norm(sim_base) + 1e-8)
                            base_i_norm = base_i / (np.linalg.norm(base_i) + 1e-8)

                            # 投影分量
                            projection = np.dot(base_i_norm, sim_norm) * sim_norm

                            # 调整：减去相似部分，增加正交性
                            adjustment = -projection * (sim - self.threshold)

                            # 应用调整到正交分量部分
                            ortho_start = self.base_dim
                            adjusted[i, ortho_start:] += adjustment[:self.ortho_dim]

                            # 重新归一化
                            norm = np.linalg.norm(adjusted[i, ortho_start:])
                            if norm > 0:
                                adjusted[i, ortho_start:] /= norm

        return adjusted

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


class OrthogonalAPITrainer:
    def __init__(self, orthogonalizer: APIEmbeddingOrthogonalizer):
        self.orthogonalizer = orthogonalizer

    def train_with_contrastive_loss(self,
                                    new_api_embeddings: np.ndarray,
                                    positive_pairs: List[Tuple[int, int]],
                                    negative_pairs: List[Tuple[int, int]],
                                    learning_rate: float = 0.001,
                                    epochs: int = 100):
        """
        使用对比损失微调新API嵌入
        保持旧API嵌入不变
        """
        import torch.optim as optim

        # 转换为可训练的Tensor
        embeddings_tensor = torch.tensor(new_api_embeddings, requires_grad=True)

        optimizer = optim.Adam([embeddings_tensor], lr=learning_rate)

        for epoch in range(epochs):
            optimizer.zero_grad()

            # 对比损失
            pos_loss = 0
            neg_loss = 0

            # 正样本对损失（拉近）
            for i, j in positive_pairs:
                sim = torch.dot(embeddings_tensor[i], embeddings_tensor[j])
                pos_loss += torch.exp(-sim)

            # 负样本对损失（推远）
            for i, j in negative_pairs:
                sim = torch.dot(embeddings_tensor[i], embeddings_tensor[j])
                neg_loss += torch.exp(sim)

            # 正交性损失
            ortho_loss = 0
            n = embeddings_tensor.shape[0]
            for i in range(n):
                for j in range(i + 1, n):
                    # 计算基础维度的相似度（应该较低）
                    sim_base = torch.dot(
                        embeddings_tensor[i, :self.orthogonalizer.base_dim],
                        embeddings_tensor[j, :self.orthogonalizer.base_dim]
                    )
                    ortho_loss += torch.relu(sim_base - 0.3)  # 希望基础相似度<0.3

            total_loss = (pos_loss / len(positive_pairs) +
                          neg_loss / len(negative_pairs) +
                          ortho_loss * 0.1)

            total_loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss.item():.4f}")

        return embeddings_tensor.detach().numpy()

