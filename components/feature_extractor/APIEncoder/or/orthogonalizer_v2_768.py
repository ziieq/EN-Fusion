import numpy as np
from typing import List, Tuple, Dict, Optional
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')


class BatchAPIOrthogonalizer:
    """
    批量API嵌入正交化器

    核心思想：使用改进的Gram-Schmidt过程，在保持总体语义结构的前提下，
    让相似API的嵌入方向尽可能正交，同时不显著改变不相似API的关系。
    """

    def __init__(self,
                 similarity_threshold: float = 0.7,
                 orthogonal_strength: float = 0.6,
                 min_semantic_preservation: float = 0.8,
                 method: str = 'adaptive'):
        """
        参数:
            similarity_threshold: 相似度阈值，高于此值的API对将被正交化
            orthogonal_strength: 正交化强度 (0-1)，1表示完全正交化
            min_semantic_preservation: 最小语义保持度 (0-1)，控制与原始嵌入的最大偏离
            method: 正交化方法
                'adaptive' - 自适应正交化（推荐）
                'full' - 完全正交化
                'selective' - 选择性正交化
        """
        assert 0 <= orthogonal_strength <= 1, "orthogonal_strength必须在0-1之间"
        assert 0 <= min_semantic_preservation <= 1, "min_semantic_preservation必须在0-1之间"

        self.sim_threshold = similarity_threshold
        self.ortho_strength = orthogonal_strength
        self.semantic_preserve = min_semantic_preservation
        self.method = method

        # 用于存储原始嵌入和相似度信息
        self.original_embeddings = None
        self.api_names = None

    def compute_pairwise_similarity(self, embeddings: np.ndarray) -> np.ndarray:
        """
        计算所有API嵌入之间的余弦相似度矩阵

        参数:
            embeddings: [n, d] 嵌入矩阵

        返回:
            similarity_matrix: [n, n] 相似度矩阵
        """
        n = embeddings.shape[0]

        # 归一化所有嵌入
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings_norm = embeddings / (norms + 1e-8)

        # 计算余弦相似度矩阵
        similarity_matrix = np.dot(embeddings_norm, embeddings_norm.T)

        # 确保对角线为1（自相似度）
        np.fill_diagonal(similarity_matrix, 1.0)

        return similarity_matrix

    def identify_similar_groups(self, similarity_matrix: np.ndarray) -> List[List[int]]:
        """
        识别需要正交化的相似API组

        参数:
            similarity_matrix: 相似度矩阵

        返回:
            groups: 相似API组列表，每个组内的API彼此高度相似
        """
        n = similarity_matrix.shape[0]
        visited = [False] * n
        groups = []

        for i in range(n):
            if not visited[i]:
                # 找到与i相似的所有API
                similar_indices = [i]
                for j in range(n):
                    if i != j and similarity_matrix[i, j] > self.sim_threshold:
                        similar_indices.append(j)

                # 如果组内不止一个API，记录这个组
                if len(similar_indices) > 1:
                    groups.append(similar_indices)
                    for idx in similar_indices:
                        visited[idx] = True
                else:
                    visited[i] = True

        return groups

    def adaptive_orthogonalize(self, embeddings: np.ndarray) -> np.ndarray:
        """
        自适应正交化方法

        核心算法：对于每个相似API组，在组内进行正交化，同时保持组间关系
        """
        n, d = embeddings.shape

        # 1. 计算相似度矩阵
        similarity_matrix = self.compute_pairwise_similarity(embeddings)

        # 2. 识别相似API组
        groups = self.identify_similar_groups(similarity_matrix)

        print(f"识别到 {len(groups)} 个相似API组")
        for i, group in enumerate(groups):
            print(f"  组{i}: {len(group)} 个API (索引: {group})")

        # 如果没有相似API组，直接返回原始嵌入
        if not groups:
            print("没有发现需要正交化的相似API组")
            return embeddings

        # 3. 对每个组进行正交化
        orthogonalized = embeddings.copy()

        for group_indices in groups:
            if len(group_indices) <= 1:
                continue

            # 提取该组的嵌入
            group_embeddings = embeddings[group_indices, :]

            # 计算组的中心方向
            group_center = np.mean(group_embeddings, axis=0)
            group_center = group_center / np.linalg.norm(group_center)

            # 计算组内各API与中心的相似度
            center_similarities = []
            for emb in group_embeddings:
                emb_norm = emb / np.linalg.norm(emb)
                sim = np.dot(emb_norm, group_center)
                center_similarities.append(sim)

            # 对组内嵌入进行Gram-Schmidt正交化（改进版）
            ortho_group_embeddings = self._improved_gram_schmidt(
                group_embeddings,
                center_direction=group_center,
                strength=self.ortho_strength
            )

            # 放回正交化后的嵌入
            for idx, emb_idx in enumerate(group_indices):
                orthogonalized[emb_idx, :] = ortho_group_embeddings[idx]

        # 4. 确保语义保持：限制与原始嵌入的最大偏离
        final_embeddings = self._enforce_semantic_preservation(
            original=embeddings,
            orthogonalized=orthogonalized,
            max_deviation=1.0 - self.semantic_preserve
        )

        return final_embeddings

    def _improved_gram_schmidt(self,
                               vectors: np.ndarray,
                               center_direction: np.ndarray,
                               strength: float = 0.7) -> np.ndarray:
        """
        改进的Gram-Schmidt正交化

        传统Gram-Schmidt会使所有向量完全正交，破坏语义信息。
        改进版：只进行部分正交化，保持与中心方向的相关性。
        """
        n, d = vectors.shape
        orthogonalized = np.zeros_like(vectors)

        # 归一化所有向量
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors_norm = vectors / (norms + 1e-8)

        # 中心方向归一化
        center_norm = center_direction / np.linalg.norm(center_direction)

        # 对每个向量进行正交化
        for i in range(n):
            # 从原始向量开始
            v = vectors_norm[i].copy()

            # 计算与之前向量的相似度
            for j in range(i):
                # 获取之前已经正交化的向量
                u = orthogonalized[j]
                u_norm = u / np.linalg.norm(u)

                # 计算投影
                proj_coeff = np.dot(v, u_norm)

                # 只减去部分投影，控制正交化程度
                # 相似度越高，减去越多
                if abs(proj_coeff) > 0.3:  # 只对相似度较高的进行处理
                    v = v - strength * proj_coeff * u_norm

            # 确保与中心方向保持一定相关性
            # 计算当前向量与中心方向的点积
            center_dot = np.dot(v, center_norm)

            # 如果与中心方向太偏离，调整回来
            min_center_similarity = 0.3  # 最小与中心方向的相似度
            if center_dot < min_center_similarity:
                # 向中心方向调整
                adjustment = (min_center_similarity - center_dot) * center_norm
                v = v + adjustment * (1 - strength)  # 调整量与正交化强度成反比

            # 归一化并存储
            v_norm = v / (np.linalg.norm(v) + 1e-8)
            orthogonalized[i] = v_norm

        # 恢复原始幅度
        orthogonalized = orthogonalized * norms

        return orthogonalized

    def _enforce_semantic_preservation(self,
                                       original: np.ndarray,
                                       orthogonalized: np.ndarray,
                                       max_deviation: float = 0.3) -> np.ndarray:
        """
        确保语义保持：限制正交化后嵌入与原始嵌入的最大偏离

        参数:
            max_deviation: 最大允许偏离（余弦相似度下降的最大值）
        """
        n = original.shape[0]
        final = orthogonalized.copy()

        for i in range(n):
            orig_norm = original[i] / (np.linalg.norm(original[i]) + 1e-8)
            ortho_norm = orthogonalized[i] / (np.linalg.norm(orthogonalized[i]) + 1e-8)

            # 计算相似度
            sim = np.dot(orig_norm, ortho_norm)

            # 如果相似度低于阈值，调整回来
            min_sim = 1.0 - max_deviation
            if sim < min_sim:
                # 计算调整方向：向原始方向移动
                adjustment_dir = orig_norm - ortho_norm * sim
                adjustment_dir = adjustment_dir / (np.linalg.norm(adjustment_dir) + 1e-8)

                # 计算需要调整的量
                needed_sim_increase = min_sim - sim

                # 应用调整
                adjusted_norm = ortho_norm + needed_sim_increase * adjustment_dir
                adjusted_norm = adjusted_norm / np.linalg.norm(adjusted_norm)

                # 恢复原始幅度
                final[i] = adjusted_norm * np.linalg.norm(orthogonalized[i])

        return final

    def full_orthogonalize(self, embeddings: np.ndarray) -> np.ndarray:
        """
        完全正交化方法：使所有嵌入都尽可能正交

        警告：这会严重破坏语义结构，只用于特定场景
        """
        n, d = embeddings.shape

        # 使用QR分解进行完全正交化
        Q, R = np.linalg.qr(embeddings.T)  # 注意：QR分解需要d >= n，否则需要调整

        # Q是正交基矩阵 [d, n]
        # 将原始嵌入投影到正交基上
        orthogonalized = Q.T @ embeddings

        # 由于QR分解会改变向量方向，我们需要调整以保持一定语义
        # 这里使用强度参数控制正交化程度
        result = embeddings * (1 - self.ortho_strength) + orthogonalized.T * self.ortho_strength

        # 归一化每行
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        result = result / (norms + 1e-8)

        # 恢复原始幅度
        original_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        result = result * original_norms

        return result

    def selective_pairwise_orthogonalize(self, embeddings: np.ndarray) -> np.ndarray:
        """
        选择性成对正交化：只对高度相似的API对进行处理

        更精细的控制，对每个相似对单独处理
        """
        n, d = embeddings.shape

        # 计算相似度矩阵
        similarity_matrix = self.compute_pairwise_similarity(embeddings)

        # 找出所有需要处理的相似对
        similar_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                if similarity_matrix[i, j] > self.sim_threshold:
                    similar_pairs.append((i, j, similarity_matrix[i, j]))

        print(f"识别到 {len(similar_pairs)} 个需要正交化的API对")

        orthogonalized = embeddings.copy()

        # 对每个相似对进行正交化处理
        for i, j, sim in similar_pairs:
            # 提取两个向量
            v1 = orthogonalized[i].copy()
            v2 = orthogonalized[j].copy()

            # 归一化
            v1_norm = v1 / np.linalg.norm(v1)
            v2_norm = v2 / np.linalg.norm(v2)

            # 当前相似度
            current_sim = np.dot(v1_norm, v2_norm)

            # 目标相似度：按正交化强度降低
            target_sim = current_sim * (1 - self.ortho_strength)

            # 如果目标相似度已经低于当前，进行调整
            if target_sim < current_sim:
                # 计算调整方向
                # 方法：将v2向垂直于v1的方向旋转
                # 投影分量
                proj = current_sim * v1_norm

                # 正交分量
                ortho = v2_norm - proj
                ortho_norm = np.linalg.norm(ortho)

                if ortho_norm > 1e-8:
                    ortho_unit = ortho / ortho_norm

                    # 计算需要旋转的角度
                    # cosθ = 目标相似度
                    target_angle = np.arccos(np.clip(target_sim, -1.0, 1.0))
                    current_angle = np.arccos(np.clip(current_sim, -1.0, 1.0))

                    # 旋转v2
                    # 新向量 = cosθ * v1方向 + sinθ * 正交方向
                    # 但注意保持v2与原始v2的一定相似度

                    # 简化方法：线性插值
                    alpha = self.ortho_strength
                    new_v2_dir = (1 - alpha) * v2_norm + alpha * ortho_unit
                    new_v2_dir = new_v2_dir / np.linalg.norm(new_v2_dir)

                    # 更新v2
                    orthogonalized[j] = new_v2_dir * np.linalg.norm(v2)

        return orthogonalized

    def orthogonalize(self,
                      embeddings: np.ndarray,
                      api_names: Optional[List[str]] = None) -> np.ndarray:
        """
        主方法：对API嵌入进行正交化

        参数:
            embeddings: [n, d] API嵌入矩阵
            api_names: 可选的API名称列表，用于调试输出

        返回:
            orthogonalized_embeddings: 正交化后的嵌入矩阵
        """
        n, d = embeddings.shape
        self.original_embeddings = embeddings.copy()
        self.api_names = api_names if api_names else [f"API_{i}" for i in range(n)]

        print(f"开始正交化处理: {n}个API, {d}维嵌入")
        print(f"参数: 相似度阈值={self.sim_threshold}, 正交化强度={self.ortho_strength}")

        # 计算原始相似度统计
        orig_sim_matrix = self.compute_pairwise_similarity(embeddings)
        orig_similarities = []
        for i in range(n):
            for j in range(i + 1, n):
                orig_similarities.append(orig_sim_matrix[i, j])

        print(f"原始嵌入平均相似度: {np.mean(orig_similarities):.4f}")
        print(f"原始嵌入相似度标准差: {np.std(orig_similarities):.4f}")

        # 根据选择的方法进行正交化
        if self.method == 'adaptive':
            result = self.adaptive_orthogonalize(embeddings)
        elif self.method == 'full':
            result = self.full_orthogonalize(embeddings)
        elif self.method == 'selective':
            result = self.selective_pairwise_orthogonalize(embeddings)
        else:
            raise ValueError(f"未知的正交化方法: {self.method}")

        # 计算正交化后的相似度统计
        ortho_sim_matrix = self.compute_pairwise_similarity(result)
        ortho_similarities = []
        for i in range(n):
            for j in range(i + 1, n):
                ortho_similarities.append(ortho_sim_matrix[i, j])

        print(f"\n正交化结果:")
        print(f"  正交化后平均相似度: {np.mean(ortho_similarities):.4f}")
        print(f"  相似度降低: {np.mean(orig_similarities) - np.mean(ortho_similarities):.4f}")
        print(
            f"  降低百分比: {(np.mean(orig_similarities) - np.mean(ortho_similarities)) / np.mean(orig_similarities) * 100:.1f}%")

        # 显示最显著的5个变化
        self._show_top_changes(orig_sim_matrix, ortho_sim_matrix)

        return result

    def _show_top_changes(self,
                          orig_sim_matrix: np.ndarray,
                          ortho_sim_matrix: np.ndarray):
        """显示相似度变化最大的API对"""
        n = orig_sim_matrix.shape[0]

        # 收集所有变化
        changes = []
        for i in range(n):
            for j in range(i + 1, n):
                change = orig_sim_matrix[i, j] - ortho_sim_matrix[i, j]
                changes.append((i, j, change))

        # 按变化量排序
        changes.sort(key=lambda x: x[2], reverse=True)

        print(f"\n相似度降低最多的API对 (前5个):")
        for i, j, change in changes[:5]:
            orig_sim = orig_sim_matrix[i, j]
            ortho_sim = ortho_sim_matrix[i, j]
            api1 = self.api_names[i]
            api2 = self.api_names[j]
            print(f"  {api1} - {api2}: {orig_sim:.4f} → {ortho_sim:.4f} (降低{change:.4f})")

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