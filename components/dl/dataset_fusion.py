from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle
import torch
torch.set_printoptions(profile="full")

class EndDataset(Dataset):
    def __init__(self, id_list_path, data_x_path, data_y_path, bi=False, device='cuda:0'):

        with open(data_x_path, 'rb') as f_x, open(data_y_path, 'rb') as f_y, open(id_list_path, 'rb') as f_id:
            self.data_x = pickle.load(f_x)
            self.data_y = pickle.load(f_y)
            self.id_list = pickle.load(f_id)

        self.data_y = torch.tensor(self.data_y)
        self.device = device
        if bi:
            self.data_y = torch.where(self.data_y > 1, 1, self.data_y)

    def __getitem__(self, idx):
        """
        返回第idx条数据。通常将label与features分开返回。
        :param idx: 数据编号
        :return: 第idx条数据
        """
        return self.id_list[idx], self.data_y[idx].to(self.device), (self.data_x[idx].to(self.device), -1)

    def __len__(self):
        """
        返回数据集中的数据总条数
        :return: 数据总条数
        """
        return self.data_y.shape[0]


class NetDataset(Dataset):
    def __init__(self, id_list_path, data_x_path, data_y_path, device='cuda:0', bi=False, f_len=200, s_len=20):
        """
        sample1: [sequence, sequence, sequence]
        sample2: [sequence, sequence, sequence]
        sample3: [sequence, sequence, sequence]
        """
        with open(data_x_path, 'rb') as f:
            self.data_x = pickle.load(f)

        with open(data_y_path, 'rb') as f:
            self.data_y = pickle.load(f)

        with open(id_list_path, 'rb') as f_id:
            self.id_list = pickle.load(f_id)

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
        print(rf'k: {f_len}')
        self.data_x = [x[:f_len] if len(x) >= f_len else x + [[[0]*25] * s_len] * (f_len-len(x)) for x in self.data_x]

        # 限制流序列长度
        print(rf'n: {s_len}')
        for x in self.data_x:
            for i in range(len(x)):
                x[i] = x[i][:s_len] if len(x[i]) >= s_len else x[i] + [[0]*25] * (s_len - len(x[i]))

        self.data_x = torch.tensor(self.data_x, dtype=torch.float32)
        self.data_y = torch.tensor(self.data_y)
        self.device = device

        if bi:
            self.data_y = torch.where(self.data_y > 1, 1, self.data_y)
        # batch * f_len * s_len * 2
        # print(self.data_x.shape)
        # print(self.data_y.shape)

    def __getitem__(self, idx):
        """
        返回第idx条数据。通常将label与features分开返回。
        :param idx: 数据编号
        :return: 第idx条数据
        """
        return self.id_list[idx], self.data_y[idx].to(self.device), (-1, self.data_x[idx].to(self.device))

    def __len__(self):
        """
        返回数据集中的数据总条数
        :return: 数据总条数
        """
        return len(self.data_y)


def process_net_feature(data_net, f_len=200, s_len=100, device='cuda:0'):
    print(rf'k: {f_len}')

    data_net_new = []
    for x in data_net:
        if x is not None:
            x = x[:f_len] if len(x) >= f_len else x + [[[0] * 25] * s_len] * (f_len - len(x))
            data_net_new.append(x)
        else:
            data_net_new.append([[[-1] * 25] * s_len] * (f_len))
    data_net = data_net_new

    # 限制流序列长度
    print(rf'n: {s_len}')
    for x in data_net:
        if x is None:
            continue
        for i in range(len(x)):
            x[i] = x[i][:s_len] if len(x[i]) >= s_len else x[i] + [[0] * 25] * (s_len - len(x[i]))

    return torch.tensor(data_net, dtype=torch.float32).to(device)


class FusionDataset(Dataset):
    def __init__(self, id_list_path, data_x_path, data_y_path, bi=False, device='cuda:0', f_len=200, s_len=20):
        # [[end, net], [end, net], [end, net], [end, net], ...]
        with open(data_x_path, 'rb') as f:
            self.data_x = pickle.load(f)

        with open(data_y_path, 'rb') as f:
            self.data_y = pickle.load(f)

        with open(id_list_path, 'rb') as f_id:
            self.id_list = pickle.load(f_id)

        self.new_data_x = []
        self.new_y = []
        for i, features in enumerate(self.data_x):
            self.new_data_x.append(features)
            self.new_y.append(self.data_y[i])

        self.data_x = self.new_data_x
        self.data_y = self.new_y
        self.data_end = [x[0] for x in self.data_x]
        for x in self.data_end:
            x.edge_index = torch.tensor(x.edge_index, dtype=torch.long)

        # data_net #########################################
        self.data_net = [x[1] for x in self.data_x]
        self.data_net = process_net_feature(self.data_net, f_len, s_len, device='cuda:0')
        self.data_y = torch.tensor(self.data_y)
        if bi:
            self.data_y = torch.where(self.data_y > 1, 1, self.data_y)

        self.device = device


    def __getitem__(self, idx):
        """
        返回第idx条数据。通常将label与features分开返回。
        :param idx: 数据编号
        :return: 第idx条数据
        """
        return self.id_list[idx], self.data_y[idx].to(self.device), (self.data_end[idx].to(self.device), self.data_net[idx].to(self.device))

    def __len__(self):
        """
        返回数据集中的数据总条数
        :return: 数据总条数
        """
        return len(self.data_y)


class MalDataset(Dataset):
    def __init__(self, id_list_path, data_x_path, data_y_path, bi=False, data_type='end', device='cuda:0', f_len=200, s_len=20):
        self.data_class = None
        if data_type == 'end':
            self.data_class = EndDataset(id_list_path, data_x_path, data_y_path, bi=bi, device=device)
        elif data_type == 'net':
            self.data_class = NetDataset(id_list_path, data_x_path, data_y_path, bi=bi, device=device, f_len=f_len, s_len=s_len)
        elif data_type == 'fusion':
            self.data_class = FusionDataset(id_list_path, data_x_path, data_y_path, bi=bi, device=device, f_len=f_len, s_len=s_len)

        print(rf'Mal dataset len:{self.data_class.__len__()}')
    def __getitem__(self, idx):
        """
        返回第idx条数据。通常将label与features分开返回。
        :param idx: 数据编号
        :return: 第idx条数据
        """
        return self.data_class.__getitem__(idx)

    def __len__(self):
        """
        返回数据集中的数据总条数
        :return: 数据总条数
        """
        return self.data_class.__len__()


if __name__ == '__main__':
    id_path = r'E:\EngineeringWare\Development\PyCharm\Project\P4_Fusion_v2\ENFusion-v3\test_data\metadata\dataset_train\E\id_train.pkl'
    data_x_path = r'E:\EngineeringWare\Development\PyCharm\Project\P4_Fusion_v2\ENFusion-v3\test_data\metadata\dataset_train\E\x_train.pkl'
    data_y_path = r'E:\EngineeringWare\Development\PyCharm\Project\P4_Fusion_v2\ENFusion-v3\test_data\metadata\dataset_train\E\y_train.pkl'

    dataset = MalDataset(id_path, data_x_path, data_y_path, bi=True, data_type='end', device='cuda:0')

    print(dataset[0])
    # for i in range(50):
    #     print(dataset.__getitem__(i))

