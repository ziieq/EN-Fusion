from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, precision_recall_curve, auc, accuracy_score
import numpy as np
from sklearn.metrics import confusion_matrix
from .txt_analyzer import generate_models_data_from_txt, multi_class_to_binary
from .auroc_prauc_computer import auroc_prauc
from .fpr_computer import macro_fpr
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from scipy import stats
from tqdm import tqdm  # 可选，用于显示自助法重抽样进度

def t_confidence_interval(scores, confidence=0.95):
    """
    t区间法计算置信区间（小样本，近似正态分布）
    :param scores: 指标序列（np.array）
    :param confidence: 置信水平（默认0.95）
    :return: 均值、置信下限、置信上限
    """
    n = len(scores)
    # if n < 2:
    #     raise ValueError("样本量过小，无法计算t区间置信区间")
    mean = np.mean(scores)
    std = np.std(scores, ddof=1)  # 样本标准差（无偏估计）
    # 计算t临界值（自由度df=n-1）
    t_critical = stats.t.ppf((1 + confidence) / 2, df=n-1)
    # 计算边际误差
    margin_error = t_critical * (std / np.sqrt(n))
    # 置信区间
    ci_lower = mean - margin_error
    ci_upper = mean + margin_error
    return mean, std, margin_error



def build_metric_dict(y_true_list, y_pred_list, y_score_list):
    metric_dict = {}
    f1_per_class_list = []
    for i in range(len(y_true_list)):
        accuracy = accuracy_score(np.array(y_true_list[i]),  np.array(y_pred_list[i]))

        f1_macro = f1_score(np.array(y_true_list[i]),  np.array(y_pred_list[i]), average='macro')
        f1_micro = f1_score(np.array(y_true_list[i]),  np.array(y_pred_list[i]), average='micro')

        auroc, prauc = auroc_prauc(y_true_list[i], y_score_list[i])

        fpr = macro_fpr(y_true_list[i], y_pred_list[i])

        metric_dict.setdefault('accuracy', []).append(accuracy)
        metric_dict.setdefault('f1_macro', []).append(f1_macro)
        metric_dict.setdefault('f1_micro', []).append(f1_micro)
        metric_dict.setdefault('auroc', []).append(auroc)
        metric_dict.setdefault('prauc', []).append(prauc)
        metric_dict.setdefault('fpr', []).append(fpr)

        f1_per_class = f1_score(y_true_list[i], y_pred_list[i], average=None)
        f1_per_class_list.append(f1_per_class)
    return metric_dict


def compute_metrics(txt_path):
    metrix_score_list, y_true_list, y_pred_list, y_score_list, matrix_list = generate_models_data_from_txt(txt_path)
    metric_dict = build_metric_dict(y_true_list, y_pred_list, y_score_list)


    print(rf'Total {len(metric_dict['accuracy'])} pieces of data.')
    for key, val in metric_dict.items():
        mean, std, margin_error = t_confidence_interval(val, confidence=0.95)
        print(rf'{key}: {mean * 100} +- str:{std * 100} +- margin_error:{margin_error * 100}')

if __name__ == '__main__':
    txt_path = r'E:\EngineeringWare\Development\PyCharm\Project\P4_Fusion_v2\m1_end\detector\Avast_random_white_128_output10_epoch20_lr0.001_testsize0.4.txt'
    metrix_score_list, y_true_list, y_pred_list, y_score_list, matrix_list = generate_models_data_from_txt(txt_path)
    metric_dict = build_metric_dict(y_true_list, y_pred_list, y_score_list)

    print(len(metric_dict['accuracy']))
    for key, val in metric_dict.items():
        mean, std, margin_error = t_confidence_interval(val, confidence=0.95)
        print(rf'{key}: {mean*100} +- str:{std*100} +- margin_error:{margin_error*100}')