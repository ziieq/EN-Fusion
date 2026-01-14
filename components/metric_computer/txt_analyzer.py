import numpy as np
import torch

def generate_models_data_from_txt(txt):
    with open(txt, 'r') as f:
        # score, y_true, y_pred, y_score, matrix
        content = f.read()
        content_list = content.split('\n')
        metrix_score_list = []
        y_true_list = []
        y_pred_list = []
        y_score_list = []
        matrix_list = []
        type_i = 0
        for i, item in enumerate(content_list):
            if len(item) == 0:
                continue
            type_i += 1
            if type_i == 1:
                metrix_score_list.append(item)
            if type_i == 2:
                y_true_list.append(np.array(eval(item)))
            elif type_i == 3:
                y_pred_list.append(np.array(eval(item)))
            elif type_i == 4:
                y_score_list.append(np.array(torch.softmax(torch.tensor(eval(item)), dim=1).numpy()))
                type_i = 0

        return metrix_score_list, y_true_list, y_pred_list, y_score_list, matrix_list


def multi_class_to_binary(y_true_list, y_pred_list, y_score_list):
    y_true_list_bi = []
    y_pred_list_bi = []
    y_score_list_bi = []
    for sub_list in y_true_list:
        y_true_list_bi.append(np.array([0 if x == 0 else 1 for x in sub_list.tolist()]))
    for sub_list in y_pred_list:
        y_pred_list_bi.append(np.array([0 if x == 0 else 1 for x in sub_list.tolist()]))
    for sub_list in y_score_list:
        y_score_list_bi.append(np.array([1-x[0] for x in sub_list]))
    return y_true_list_bi, y_pred_list_bi, y_score_list_bi

if __name__ == '__main__':
    txt_path = r'USTC_output11_epoch50.txt'
    metrix_score_list, y_true_list, y_pred_list, y_score_list, matrix_list = generate_models_data_from_txt(txt_path)
    # print(len(metrix_score_list))
    # print(len(y_true_list))
    # print(len(y_pred_list))
    # print(len(y_score_list))
    y_true_list_bi, y_pred_list_bi, y_score_list_bi = multi_class_to_binary(y_true_list, y_pred_list, y_score_list)
    print(y_score_list_bi[0])