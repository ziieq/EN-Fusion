import numpy as np
from sklearn.metrics import confusion_matrix



def macro_fpr(y_true, y_pred):
    # # 1. 模拟数据
    # y_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])  # 3类（0,1,2）
    # y_pred = np.array([0, 0, 1, 1, 1, 2, 2, 0, 2])

    # 2. 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    K = cm.shape[0]  # 类别数
    total_samples = np.sum(cm)

    # 3. 逐类计算FPR
    fpr_per_class = []
    for i in range(K):
        fp = sum(cm[j, i] for j in range(K) if j != i)  # 其他类预测为i的数量
        tn = total_samples - sum(cm[i, :]) - sum(cm[:, i]) + cm[i, i]  # 推导公式
        fpr = fp / (fp + tn) if (fp + tn) != 0 else 0
        fpr_per_class.append(fpr)

    # 4. 计算宏平均和加权平均FPR
    macro_fpr = np.mean(fpr_per_class)
    weighted_fpr = np.average(fpr_per_class, weights=np.sum(cm, axis=1))

    # print(f"逐类FPR: {fpr_per_class}")
    # print(f"宏平均FPR: {macro_fpr:.4f}")
    # print(f"加权平均FPR: {weighted_fpr:.4f}")
    return macro_fpr