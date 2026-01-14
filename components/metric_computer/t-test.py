import scipy.stats as stats
import numpy as np
from txt_analyzer import generate_models_data_from_txt, multi_class_to_binary
from metric_computer import build_metric_dict

def paired_ttest(x1, x2):
    """
    配对样本T检验（适用于：同一测试集/样本集上的两组数据对比，如我们的方法vs基线方法的性能值）
    参数：
        x1: 第一组数据（数组/列表类型，如我们的方法的5折交叉验证准确率）
        x2: 第二组数据（数组/列表类型，长度需与x1一致，如某基线方法的5折交叉验证准确率）
    返回：
        t_value: T检验的t统计量（保留4位小数）
        p_value: 对应的p值（保留4位小数）
    异常处理：
        两组数据长度不一致时，抛出 ValueError
    """
    # 类型转换与长度校验
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    if len(x1) != len(x2):
        raise ValueError("错误：两组数据长度必须一致（配对样本要求一一对应）")

    # 执行配对样本T检验
    t_stat, p_stat = stats.ttest_rel(x1, x2)

    # 保留4位小数，提升结果可读性（适配SCI论文表格呈现）
    return round(t_stat, 4), round(p_stat, 4)

if __name__ == '__main__':
    txt_path = r'E:\EngineeringWare\Development\PyCharm\Project\P4_Fusion_v2\statistc\res\MalSE\avast_family_epoch0_lr0.txt'
    metrix_score_list, y_true_list, y_pred_list, y_score_list, matrix_list = generate_models_data_from_txt(txt_path)
    metric_dict1 = build_metric_dict(y_true_list, y_pred_list, y_score_list)

    txt_path = r'E:\EngineeringWare\Development\PyCharm\Project\P4_Fusion_v2\statistc\res\TSMal\Avast_output10_epoch200.txt'
    metrix_score_list, y_true_list, y_pred_list, y_score_list, matrix_list = generate_models_data_from_txt(txt_path)
    metric_dict2 = build_metric_dict(y_true_list, y_pred_list, y_score_list)

    txt_path = r'E:\EngineeringWare\Development\PyCharm\Project\P4_Fusion_v2\statistc\res\Ablation\端侧LLM消融\test04\Avast_llm_white_128_output10_epoch20.txt'
    metrix_score_list, y_true_list, y_pred_list, y_score_list, matrix_list = generate_models_data_from_txt(txt_path)
    metric_dict3 = build_metric_dict(y_true_list, y_pred_list, y_score_list)

    t, p = paired_ttest(metric_dict1['f1_macro'], metric_dict3['f1_macro'])
    print(t, p)

    t, p = paired_ttest(metric_dict2['f1_macro'], metric_dict3['f1_macro'])
    print(t, p)