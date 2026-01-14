from torch.utils.data import Dataset
import pandas as pd
import pickle
import torch
from torch.nn.utils.rnn import pad_sequence
import seaborn as sns
import matplotlib.pyplot as plt
torch.set_printoptions(profile="full")
from collections import Counter
import numpy as np


"""
sample1: [[sequence, service, port_range], [sequence, service, port_range], [sequence, service, port_range]]
sample2: [[sequence, service, port_range], [sequence, service, port_range], [sequence, service, port_range]]
sample3: [[sequence, service, port_range], [sequence, service, port_range], [sequence, service, port_range]]
"""

def oversample_multiclass_to_average(x: torch.Tensor, y: torch.Tensor, flowcnt):
    """
    多类别（单标签）场景下，对每个类别过采样至所有类别的平均样本数量
    Args:
        x: 特征张量 (N, ...)，N为样本数，后续维度为特征维度（如C,H,W）
        y: 类别标签张量 (N,)，单标签，值为类别索引（如0,1,2...）
    Returns:
        balanced_x: 平衡后的特征张量
        balanced_y: 平衡后的类别标签张量
    """
    # 输入校验
    if len(y.shape) != 1:
        raise ValueError(f"多类别单标签张量y应为1维 (N,)，当前形状：{y.shape}")
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"x和y的样本数不匹配：x={x.shape[0]}, y={y.shape[0]}")

    # 步骤1：统计每个类别的样本数量
    y_np = y.cpu().numpy()
    label_counts = Counter(y_np)  # {类别索引: 样本数}
    classes = sorted(label_counts.keys())  # 按类别索引排序
    class_nums = np.array([label_counts[c] for c in classes])  # 各类别样本数数组
    avg_num = np.mean(class_nums)  # 所有类别的平均样本数量

    print(f"原始类别样本分布：{label_counts}")
    print(f"类别平均样本数量：{avg_num:.2f}")

    # 初始化平衡后的数据集（先保留原始数据）
    balanced_x = [x.clone()]
    balanced_y = [y.clone()]
    balanced_flowcnt = [flowcnt.clone()]
    # 步骤2：遍历每个类别，对样本数低于平均值的类别进行过采样
    for cls in classes:
        cls_num = label_counts[cls]
        if cls_num >= avg_num:
            continue  # 该类别样本数已达标，跳过

        # 计算需要补充的样本数（向上取整，确保不低于平均值）
        target_num = int(np.ceil(avg_num))
        need_sample_num = target_num - cls_num
        if need_sample_num <= 0:
            continue

        # 步骤3：获取当前类别的所有样本索引
        cls_indices = torch.where(y == cls)[0]
        cls_sample_num = len(cls_indices)
        if cls_sample_num == 0:
            print(f"警告：类别{cls}无样本，跳过采样")
            continue

        # 步骤4：随机过采样当前类别的样本（可重复采样，保证多样性）
        # 随机选择需要复制的样本索引（允许重复，适配样本数不足的情况）
        sample_indices = torch.randint(0, cls_sample_num, (need_sample_num,))
        selected_indices = cls_indices[sample_indices]  # 映射到原始数据的索引

        # 提取采样的样本和标签
        sampled_x = x[selected_indices]
        sampled_y = y[selected_indices]
        sampled_flowcnt = flowcnt[selected_indices]
        # 加入平衡数据集
        balanced_x.append(sampled_x)
        balanced_y.append(sampled_y)
        balanced_flowcnt.append(sampled_flowcnt)

    # 步骤5：合并所有采样数据，并打乱顺序（避免同类样本聚集）
    balanced_x = torch.cat(balanced_x, dim=0)
    balanced_y = torch.cat(balanced_y, dim=0)
    balanced_flowcnt = torch.cat(balanced_flowcnt, dim=0)
    # 打乱数据集（保持x和y的对应关系）
    perm = torch.randperm(balanced_x.shape[0])
    balanced_x = balanced_x[perm]
    balanced_y = balanced_y[perm]
    balanced_flowcnt = balanced_flowcnt[perm]
    # 验证采样后分布
    final_counts = Counter(balanced_y.cpu().numpy())
    print(f"采样后类别样本分布：{final_counts}")
    print(f"采样后平均样本数量：{np.mean(list(final_counts.values())):.2f}")

    return balanced_x, balanced_y, balanced_flowcnt


class NetDataset(Dataset):
    def __init__(self, data_x_path, data_y_path, device='cuda:0', f_len=200, s_len=20, is_train=False):
        """
        sample1: [sequence, sequence, sequence]
        sample2: [sequence, sequence, sequence]
        sample3: [sequence, sequence, sequence]

        样本总数 * 流数 * 包数 * 25
        """
        with open(data_x_path, 'rb') as f:
            self.data_x = pickle.load(f)

        with open(data_y_path, 'rb') as f:
            self.data_y = pickle.load(f).tolist()

        # 对 y == 0 超采样
        zero_x = [xi for xi, yi in zip(self.data_x, self.data_y) if yi == 0]
        zero_y = [yi for xi, yi in zip(self.data_x, self.data_y) if yi == 0]
        self.data_x = self.data_x + zero_x
        self.data_y = self.data_y + zero_y

        s_len_ave = 0
        f_len_ave = 0
        for x in self.data_x:
            sequence_list = [len(flow_list) for flow_list in x]
            s_len_ave += sum(sequence_list) / len(sequence_list)
            f_len_ave += len(x)
        s_len_ave = int(s_len_ave / len(self.data_x))
        f_len_ave = int(f_len_ave / len(self.data_x))
        print(f_len_ave, s_len_ave)

        # 限制流数
        print(rf'f_len: {f_len}')
        self.flow_cnt = torch.tensor([len(x) for x in self.data_x])
        self.data_x = [x[:f_len] if len(x) >= f_len else x + [[[0]*25] * s_len] * (f_len-len(x)) for x in self.data_x]

        # 限制流序列长度
        print(rf's_len: {s_len}')
        for x in self.data_x:
            for i in range(len(x)):
                x[i] = x[i][:s_len] if len(x[i]) >= s_len else x[i] + [[0]*25] * (s_len - len(x[i]))
        self.data_x = torch.tensor(self.data_x, dtype=torch.float32)
        self.data_y = torch.tensor(self.data_y)
        # if is_train:
        #     self.data_x, self.data_y, self.flow_cnt = oversample_multiclass_to_average(self.data_x, self.data_y, self.flow_cnt)
        print(self.data_x.shape)
        self.device = device

        # batch * f_len * s_len * 2

    def __getitem__(self, idx):
        """
        返回第idx条数据。通常将label与features分开返回。
        :param idx: 数据编号
        :return: 第idx条数据
        """
        return self.data_y[idx].to(self.device), [-1, self.data_x[idx].to(self.device)], self.flow_cnt[idx]

    def __len__(self):
        """
        返回数据集中的数据总条数
        :return: 数据总条数
        """
        return len(self.data_x)


def show_lineplot(y):

    data = pd.DataFrame({
        'pkt_len': y[0],
        'gap': y[1],
        'service': y[2],
        'port': y[3],

    })
    plt.figure(figsize=(6, 2))

    sns.lineplot(data=data, palette='Blues_r', markers='*', sizes=1)
    ax = plt.gca()

    plt.grid(axis='y',
             color='black',
             linestyle='--',
             linewidth=0.5,
             alpha=1)

    # 添加标签和图例
    #plt.legend(loc='upper right')
    plt.legend([], [], frameon=False)
    plt.xlabel('Unit')
    plt.ylabel('Benign')
    # plt.yticks([1.1, 2.3, 3.5, 4.7], labels=['port', 'service', 'time_gap', 'pkt_len'], rotation=0)
    plt.yticks([0.6, 1.1, 1.7, 2.3, 2.9, 3.5, 4.1], labels=['port', '', 'service', '', 'gap', '', 'pkt_len'], rotation=30)

    ax = plt.gca()
    ax.yaxis.grid(True)  # 先开启所有y轴网格线
    for i, line in enumerate(ax.yaxis.get_gridlines()):
        # if line.get_ydata()[0] not in sorted(yticks)[:3]:  # 如果不是底部三条
        if i % 2 == 0:
            line.set_visible(False)

    plt.rcParams['figure.autolayout'] = False  # 关闭自动布局
    ax.tick_params(axis='both', which='both', length=0)  # 删除横纵坐标轴上的刻度线


    # plt.gca().tick_params(left=False, labelleft=False)

    # 显示图形
    plt.show()


if __name__ == '__main__':
    dataset = NetDataset('../data/EN2025_both/data_x.pkl', '../data/EN2025_both/data_y.pkl')
    dataset.__getitem__(8)
    print(dataset.__len__())



