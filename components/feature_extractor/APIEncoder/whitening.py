import numpy as np
from scipy import linalg


class Whitening:
    """
    简化版的BERT-whitening实现，专注于小样本稳定性
    参考论文: Whitening Sentence Representations for Better Semantics and Faster Retrieval
    """

    def __init__(self, n_components=None, reg_param=0.1, min_var=1e-4):
        """
        初始化白化器

        Args:
            n_components: 降维维度，None为不降维，小样本建议设为较小值
            reg_param: 正则化参数，防止小样本过拟合
            min_var: 最小方差阈值，防止数值不稳定
        """
        self.n_components = n_components
        self.reg_param = reg_param
        self.min_var = min_var

        # 内部状态
        self.W = None  # 白化矩阵
        self.mu = None  # 均值向量

    def fit(self, vecs):
        """
        拟合数据，计算白化参数

        Args:
            vecs: 向量矩阵，shape=(n_samples, n_features)
        """
        n_samples, n_features = vecs.shape

        # 1. 计算均值
        self.mu = vecs.mean(axis=0, keepdims=True)
        X_centered = vecs - self.mu

        # 2. 计算协方差矩阵（添加正则化）
        cov = np.cov(X_centered.T)

        # 小样本情况：添加对角线正则化
        if n_samples < n_features * 5:
            trace = np.trace(cov)
            cov = (1 - self.reg_param) * cov + self.reg_param * (trace / n_features) * np.eye(n_features)

        # 3. 特征分解
        eigvals, eigvecs = linalg.eigh(cov)

        # 确保特征值为正
        eigvals = np.maximum(eigvals, self.min_var)

        # 4. 确定降维维度
        if self.n_components is None:
            # 自动选择：保留解释95%方差的维度
            total_var = np.sum(eigvals)
            cum_var = np.cumsum(eigvals[::-1]) / total_var
            n_keep = np.argmax(cum_var >= 0.95) + 1
            n_keep = min(n_keep, n_samples // 2)  # 不超过样本数的一半
            n_keep = max(n_keep, min(32, n_features))  # 至少保留32维
        else:
            n_keep = min(self.n_components, len(eigvals))

        # 5. 排序并截断
        idx = eigvals.argsort()[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        eigvals_k = eigvals[:n_keep]
        eigvecs_k = eigvecs[:, :n_keep]

        # 6. 计算白化矩阵
        scaling = 1.0 / np.sqrt(eigvals_k)
        self.W = eigvecs_k @ np.diag(scaling)

        return self

    def transform(self, vecs, normalize=True):
        """
        白化转换

        Args:
            vecs: 待转换的向量
            normalize: 是否进行L2归一化

        Returns:
            白化后的向量
        """
        if self.W is None or self.mu is None:
            raise ValueError("请先调用fit方法")

        # 中心化
        X_centered = vecs - self.mu

        # 白化
        X_white = X_centered @ self.W

        # 可选归一化
        if normalize:
            norms = np.linalg.norm(X_white, axis=1, keepdims=True)
            X_white = X_white / np.maximum(norms, 1e-8)

        return X_white

    def fit_transform(self, vecs, normalize=True):
        """拟合并转换数据"""
        self.fit(vecs)
        return self.transform(vecs, normalize)

    def get_statistics(self):
        """获取统计信息"""
        if self.W is None:
            return {}

        return {
            'input_dim': self.W.shape[1] if self.W is not None else None,
            'output_dim': self.W.shape[0] if self.W is not None else None,
            'n_components': self.n_components
        }


def simple_whitening(vecs, n_components=None):
    """
    最简单直接的白化函数

    Args:
        vecs: 输入向量矩阵
        n_components: 降维维度

    Returns:
        白化后的向量
    """
    # 中心化
    mu = vecs.mean(axis=0, keepdims=True)
    X_centered = vecs - mu

    # 计算协方差
    cov = np.cov(X_centered.T)

    # 特征分解
    eigvals, eigvecs = linalg.eigh(cov)

    # 处理小特征值
    eigvals = np.maximum(eigvals, 1e-6)

    # 降维
    if n_components is not None:
        n_keep = min(n_components, len(eigvals))
        idx = eigvals.argsort()[::-1][:n_keep]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

    # 白化矩阵
    scaling = 1.0 / np.sqrt(eigvals)
    W = eigvecs @ np.diag(scaling)

    # 白化转换
    X_white = X_centered @ W

    return X_white


# 使用示例
def demo_whitening():
    """演示白化使用"""
    np.random.seed(42)

    # 生成小样本数据（模拟BERT向量）
    n_samples = 50  # 小样本
    n_features = 768
    print(f"小样本测试: {n_samples}个样本, {n_features}维")

    # 生成有偏相关的数据
    X = np.random.randn(n_samples, n_features)
    X = X @ np.random.randn(n_features, n_features) * 0.5  # 添加相关性
    X = X + np.random.randn(n_features) * 2  # 添加偏置

    print(f"\n原始数据:")
    print(f"  形状: {X.shape}")
    print(f"  均值: {X.mean(axis=0)[:3].round(3)}...")

    # 方法1: 使用完整类
    print(f"\n方法1: 使用Whitening类")
    whitener = Whitening(n_components=128, reg_param=0.1)
    X_white1 = whitener.fit_transform(X)

    stats = whitener.get_statistics()
    print(f"  输入维度: {stats['input_dim']}")
    print(f"  输出维度: {stats['output_dim']}")

    # 方法2: 使用简单函数
    print(f"\n方法2: 使用简单函数")
    X_white2 = simple_whitening(X, n_components=128)

    # 验证白化效果
    def check_whitening(vecs, name):
        cov = np.cov(vecs.T)
        diag_var = np.diag(cov)

        print(f"\n{name}效果验证:")
        print(f"  方差均值: {diag_var.mean():.4f} (理想: 1.0)")
        print(f"  方差范围: [{diag_var.min():.4f}, {diag_var.max():.4f}]")
        print(f"  协方差均值绝对值: {np.abs(cov - np.diag(diag_var)).mean():.6f}")

    check_whitening(X_white1, "方法1")
    check_whitening(X_white2, "方法2")

    return X_white1, X_white2


# 与BERT集成的简单示例
def bert_whitening_simple(bert_vectors, train_ratio=0.8):
    """
    BERT白化的简单集成

    Args:
        bert_vectors: BERT输出的所有向量
        train_ratio: 用于训练白化器的比例

    Returns:
        (训练集白化向量, 测试集白化向量, 白化器)
    """
    n_samples = len(bert_vectors)
    n_train = int(n_samples * train_ratio)

    # 分割训练测试
    train_vecs = bert_vectors[:n_train]
    test_vecs = bert_vectors[n_train:]

    print(f"数据划分: {n_train}训练, {n_samples - n_train}测试")

    # 创建白化器（小样本自动适应）
    n_components = min(256, n_train // 2) if n_train < 500 else 256
    reg_param = 0.2 if n_train < 100 else 0.1

    whitener = Whitening(n_components=n_components, reg_param=reg_param)

    # 训练并转换
    train_white = whitener.fit_transform(train_vecs)
    test_white = whitener.transform(test_vecs)

    print(f"白化后维度: {train_white.shape[1]}维")

    return train_white, test_white, whitener


if __name__ == "__main__":
    print("=" * 60)
    print("小样本BERT-whitening演示")
    print("=" * 60)

    # 运行演示
    X_white1, X_white2 = demo_whitening()

    print("\n" + "=" * 60)
    print("使用建议:")
    print("1. 样本数 < 100: 使用 Whitening(reg_param=0.2)")
    print("2. 样本数 100-500: 使用 Whitening(reg_param=0.1)")
    print("3. 样本数 > 500: 使用 Whitening(reg_param=0.05)")
    print("4. 降维维度建议: min(256, 样本数//2)")
    print("=" * 60)