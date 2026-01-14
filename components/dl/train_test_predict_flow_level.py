import os.path
import pickle
import numpy as np
import tqdm
from .model import ENFusion, GATGraphClassifier, NetGRU
import torch.nn as nn
import torch.optim as optim
import torch
from torch_geometric.data import Data, DataLoader
import torchmetrics
from .dataset_net_flow_level import NetDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


def train_single_side(model, train_set, test_set, side, output_dim=2, epochs=20, device='cuda:0'):
    print('* Start single side training：')
    # 定义损失函数、优化器、DataLoader
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-3)
    dataloader = DataLoader(dataset=train_set, batch_size=64, shuffle=True)

    task_str = 'multiclass' if output_dim == 2 else 'multiclass'
    acc = torchmetrics.Accuracy(task=task_str, num_classes=output_dim, average='micro').to(device)
    f1 = torchmetrics.F1Score(task=task_str, num_classes=output_dim, average='macro').to(device)
    precision = torchmetrics.Precision(task=task_str, num_classes=output_dim, average='macro').to(device)
    recall = torchmetrics.Recall(task=task_str, num_classes=output_dim, average='macro').to(device)

    # 开始训练
    for epoch in range(epochs):
        for labels, features, flow_cnts in tqdm.tqdm(dataloader):
            # 正向传播
            y_pre = model.forward(features, side) # , stage
            # 反向传播
            optimizer.zero_grad()
            loss = criterion(y_pre, labels)
            loss.backward()
            optimizer.step()
            # 统计指标
            y_pre = torch.argmax(y_pre, dim=1)
            acc.update(y_pre, labels)
            f1.update(y_pre, labels)
            recall.update(y_pre, labels)
            precision.update(y_pre, labels)
        # 计算所有batch的平均指标
        acc_avg = acc.compute()
        f1_avg = f1.compute()
        precision_avg = precision.compute()
        recall_avg = recall.compute()
        print(rf'  epoch: {epoch+1}, acc: {acc_avg}, f1: {f1_avg}, precision: {precision_avg}, recall: {recall_avg}')
        # 重置指标，以开始统计下一个epoch
        acc.reset()
        f1.reset()
        precision.reset()
        recall.reset()

def predict_set(model, test_set, side):
    print('*** Start testing：')
    dataloader = DataLoader(dataset=test_set, batch_size=64, shuffle=True)

    # 开始训练
    y_true_all, y_pred_all, y_score_all = [], [], []
    id_list = []
    for labels, features, flow_cnts in tqdm.tqdm(dataloader):
        y_pre = model.forward(features, side)
        y_score_list = y_pre.tolist()
        label_list = labels.tolist()
        for t in range(len(labels)):
            y_score_all += [y_score_list[t]] * flow_cnts[t]
            y_true_all += [label_list[t]] * flow_cnts[t]
        y_pre = torch.argmax(y_pre, dim=1)
        y_pre_list = y_pre.tolist()
        for t in range(len(labels)):
            y_pred_all += [y_pre_list[t]] * flow_cnts[t]

    accuracy = accuracy_score(y_true_all, y_pred_all)
    f1 = f1_score(y_true_all, y_pred_all, average='macro')
    print('*** Test set accuracy: {:.4f} f1: {:.4f}'.format(accuracy, f1))
    return id_list, y_true_all, y_pred_all, y_score_all


def split_dataset(data_dir_path, test_size=0.4):
    data_x_path, data_y_path, data_id_path = os.path.join(data_dir_path, 'x_all.pkl'), os.path.join(data_dir_path, 'y_all.pkl'), os.path.join(data_dir_path, 'id_all.pkl')
    with open(data_x_path, 'rb') as f:
        data_x = pickle.load(f)
    with open(data_y_path, 'rb') as f:
        data_y = pickle.load(f)
    with open(data_id_path, 'rb') as f:
        data_id = pickle.load(f)
        data_id = data_id[:len(data_x)]

    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(data_x, data_y, data_id, test_size=test_size, shuffle=True)
    with open(rf'{data_dir_path}/x_train.pkl', 'wb') as f:
        pickle.dump(X_train, f)
    with open(rf'{data_dir_path}/x_test.pkl', 'wb') as f:
        pickle.dump(X_test, f)
    with open(rf'{data_dir_path}/y_train.pkl', 'wb') as f:
        pickle.dump(y_train, f)
    with open(rf'{data_dir_path}/y_test.pkl', 'wb') as f:
        pickle.dump(y_test, f)
    with open(rf'{data_dir_path}/id_train.pkl', 'wb') as f:
        pickle.dump(id_train, f)
    with open(rf'{data_dir_path}/id_test.pkl', 'wb') as f:
        pickle.dump(id_test, f)


def write_output(file_path, y_true, y_pred, y_score):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    res_str = '* Test set accuracy: {:.4f} f1: {:.4f}'.format(accuracy, f1)
    with open(file_path, 'a+', encoding='utf-8') as f:
        f.write(res_str+'\n')
        f.write(str(y_true)+'\n')
        f.write(str(y_pred)+'\n')
        f.write(str(y_score) + '\n')


def run_training_flow_level(model, data_dir_path, side, output_dim, dst_model_dir_path, epoch, f_len, s_len):
    device = 'cuda:0'
    torch.cuda.init()

    # 实例化Dataset类
    train_set = NetDataset(rf'{data_dir_path}/x_train.pkl', rf'{data_dir_path}/y_train.pkl',device=device, f_len=f_len, s_len=s_len, is_train=True)
    test_set = NetDataset(rf'{data_dir_path}/x_test.pkl', rf'{data_dir_path}/y_test.pkl',device=device, f_len=f_len, s_len=s_len, is_train=False)

    # 开始训练
    if model is None:
        model = ENFusion(output_dim).to(device)

    train_single_side(model, train_set, test_set, side, output_dim=output_dim, epochs=epoch, device=device)

    # 保存模型
    torch.save(model, f'{dst_model_dir_path}/model.pth')

    # 预测结果
    id_list, y_true_all, y_pred_all, y_score_all = predict_set(model, test_set, side)
    return id_list, y_true_all, y_pred_all, y_score_all


