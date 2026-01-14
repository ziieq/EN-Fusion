import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.preprocessing import label_binarize

# 1. 定义真实标签和预测概率
y_true = np.array([0, 1, 2, 3, 4])  # 完美覆盖0-4，每个类别各1个样本

# 预测概率矩阵（shape=(5,5)，对应0-4类，每行概率和≈1，模拟"有一定误差但正确类别概率最高"的场景）
y_pred = np.array([
    # 样本1：真实0 → 0类概率最高（0.95），其余类别低概率
    [0.95, 0.01, 0.01, 0.02, 0.01],
    # 样本2：真实1 → 1类概率最高（0.93），其余类别低概率
    [0.02, 0.93, 0.01, 0.02, 0.02],
    # 样本3：真实2 → 2类概率最高（0.90），其余类别低概率（模拟轻微误差）
    [0.03, 0.04, 0.55, 0.02, 0.01],
    # 样本4：真实3 → 3类概率最高（0.88），其余类别低概率
    [0.31, 0.22, 0.13, 0.78, 0.26],
    # 样本5：真实4 → 4类概率最高（0.91），其余类别低概率
    [0.02, 0.01, 0.04, 0.02, 0.91]
])


def auroc_prauc(y_true, y_pred):
    # 2. 确定所有类别（这里y_pred的列数是10，对应类别0-9）
    classes = np.arange(y_pred.shape[1])  # 类别0,1,2,...,9

    # 3. 将真实标签二值化（OvR格式）：shape=(n_samples, n_classes)
    if len(classes) > 2:
        y_true_bin = label_binarize(y_true, classes=classes)
    else:
        y_true_bin = np.zeros((len(y_true), len(classes)))  # 初始化全0矩阵
        for idx, cls in enumerate(classes):
            y_true_bin[:, idx] = (y_true == cls).astype(int)  # 逐列赋值
    # ===================== 计算AUROC =====================
    # 方法1：宏平均AUROC（对每个类别计算AUROC后取平均）
    roc_auc_macro = roc_auc_score(y_true_bin, y_pred, average='macro')
    # 方法2：微平均AUROC（将所有样本的正负例合并后计算）
    roc_auc_micro = roc_auc_score(y_true_bin, y_pred, average='micro')


    # ===================== 计算PRAUC =====================
    def pr_auc_score(y_true_bin, y_pred, average='macro'):
        """
        计算多分类PRAUC（OvR策略）
        :param y_true_bin: 二值化后的真实标签，shape=(n_samples, n_classes)
        :param y_pred: 预测概率，shape=(n_samples, n_classes)
        :param average: 'macro'（宏平均）/'micro'（微平均）
        :return: PRAUC值
        """
        n_classes = y_true_bin.shape[1]
        pr_aucs = []

        if average == 'macro':
            # 对每个类别计算PRAUC
            for i in range(n_classes):
                # 跳过无正例的类别（避免计算错误）
                if np.sum(y_true_bin[:, i]) == 0:
                    pr_aucs.append(0.0)
                    continue
                # 计算该类别的精确率-召回率曲线
                precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred[:, i])
                # 计算PR曲线下面积
                pr_auc = auc(recall, precision)
                pr_aucs.append(pr_auc)
            # 宏平均
            return np.mean(pr_aucs)

        elif average == 'micro':
            # 微平均：合并所有类别的正负例
            precision, recall, _ = precision_recall_curve(y_true_bin.ravel(), y_pred.ravel())
            pr_auc = auc(recall, precision)
            return pr_auc
        return None

    # 计算宏平均/微平均PRAUC
    pr_auc_macro = pr_auc_score(y_true_bin, y_pred, average='macro')
    pr_auc_micro = pr_auc_score(y_true_bin, y_pred, average='micro')

    # ===================== 输出结果 =====================
    # print("=== AUROC 结果 ===")
    # print(f"宏平均AUROC: {roc_auc_macro:.4f}")
    # print(f"微平均AUROC: {roc_auc_micro:.4f}")
    # print("\n=== PRAUC 结果 ===")
    # print(f"宏平均PRAUC: {pr_auc_macro:.4f}")
    # print(f"微平均PRAUC: {pr_auc_micro:.4f}")
    return roc_auc_macro, pr_auc_macro

# 可选：输出每个类别的AUROC和PRAUC
# print("\n=== 每个类别的AUROC ===")
# for i, cls in enumerate(classes):
#     if np.sum(y_true_bin[:, i]) == 0:
#         roc_auc_cls = 0.0
#     else:
#         roc_auc_cls = roc_auc_score(y_true_bin[:, i], y_pred[:, i])
#     print(f"类别 {cls}: {roc_auc_cls:.4f}")
#
# print("\n=== 每个类别的PRAUC ===")
# for i, cls in enumerate(classes):
#     if np.sum(y_true_bin[:, i]) == 0:
#         pr_auc_cls = 0.0
#     else:
#         precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred[:, i])
#         pr_auc_cls = auc(recall, precision)
#     print(f"类别 {cls}: {pr_auc_cls:.4f}")