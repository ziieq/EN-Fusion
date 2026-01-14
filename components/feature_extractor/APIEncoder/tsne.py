import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn import datasets
from sklearn.manifold import TSNE
from matplotlib import cm
import random

def tsne_visualization(data, labels=None, perplexity=10, n_iter=5000, random_state=6):
    """
    使用t-SNE进行二维可视化

    参数:
    - data: 高维数据 (n_samples, n_features)
    - labels: 数据点的标签 (可选)
    - perplexity: t-SNE的困惑度参数
    - n_iter: 优化迭代次数
    - random_state: 随机种子
    """


    # 创建t-SNE模型
    tsne = TSNE(n_components=2,
                perplexity=perplexity,
                n_iter=n_iter,
                random_state=random.randint(0, 5000),
                init='random',  # 可以改为'pca'以获得更稳定的结果
                learning_rate='auto')

    # 执行降维
    data_2d = tsne.fit_transform(data)

    # 可视化
    plt.figure(figsize=(10, 8))
    markers = ['o', 's', '^', 'D', 'x', '*', 'p', 'h', '+', '>', '<', 'P', 'X', 'd', 'v']

    if labels is not None:
        # 如果有标签，使用不同颜色表示不同类别

        # unique_labels = np.random.permutation(np.unique(labels))
        unique_labels = np.array(['com', 'synchronization', 'crypto', 'network', 'process', 'threading', 'system',
 'filesystem', 'windows', 'hooking', 'misc', '__notification__', 'registry',
 'device', 'services'])
        print(unique_labels)

        label2marker = {label: marker for label, marker in zip(unique_labels, markers)}

        colors = cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        label2color = {}

        for label, color in zip(unique_labels, colors):
            label2color[label] = color

        color_sequence = []
        for label in labels:
            color_sequence.append(label2color[label])

        for label in unique_labels:
            tmp_data = []
            for i, d in enumerate(data_2d):
                if labels[i] != label:
                    continue
                tmp_data.append(d)
            tmp_data = np.array(tmp_data)

            plt.scatter(tmp_data[:, 0], tmp_data[:, 1], alpha=0.7, c=label2color[label], label=label, s=100, marker=label2marker[label])

    else:
        # 如果没有标签，所有点使用相同颜色
        plt.scatter(data_2d[:, 0], data_2d[:, 1], alpha=0.7)

    plt.legend(loc='upper left', ncol=2, fontsize=10, prop={'weight': 'bold'})
    # plt.title('Embedding with API Name', fontsize=28, weight='bold')
    plt.xlabel('t-SNE Dimension x', fontsize=26)
    plt.ylabel('t-SNE Dimension y', fontsize=26)
    plt.title(
        '(b) Embedding with LLM Enhancement',  # 标题文本  # (a) Embedding with API Name
        y=-0.25,  # 垂直位置：负数表示在图表下方（底部），数值可按需调整
        ha='center',  # 水平居中（默认已居中，显式设置更稳妥）
        fontsize=28,  # 可选：设置字体大小
        weight='bold'
    )

    # 4. 调整边距（避免标题被截断，关键步骤）
    plt.subplots_adjust(bottom=0.2, top=0.95)  # 增大底部边距，bottom取值0~1
    plt.show()

    return data_2d


# 示例使用：在鸢尾花数据集上应用t-SNE
if __name__ == "__main__":
    # 加载鸢尾花数据集
    iris = datasets.load_iris()
    X = iris.data  # 特征
    y = iris.target  # 标签

    print("原始数据形状:", X.shape)

    # 执行t-SNE可视化
    X_2d = tsne_visualization(X, y, perplexity=30, n_iter=1000)

    print("降维后数据形状:", X_2d.shape)