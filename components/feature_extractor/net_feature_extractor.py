import os
import shutil
import time
import uuid
import dpkt
import torch
import math
import pickle
import tqdm
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
import socket

class NetBehaviorExtractor:
    def __init__(self):
        self.service_dict = {
            22: 0,
            23: 0,
            80: 1,
            443: 1,
            53: 2,
            'other': 3,
        }

    def split_and_extract(self, src_file_path, tmp_dir_path):
        # 针对一个进程，进行切分为多个块
        pcap_sets_2k = []
        with open(src_file_path, 'rb') as f:
            pcap = dpkt.pcap.Reader(f)

            pcap_set = []
            for ts, pkt in pcap:
                if len(pcap_set) < 2000:
                    pcap_set.append([ts, pkt])
                else:
                    pcap_sets_2k.append(pcap_set)
                    pcap_set = []
            if len(pcap_set) != 0:
                pcap_sets_2k.append(pcap_set)

        process_feature_list = []
        flow_id_list_all = []

        for pcap_set in pcap_sets_2k:
            flow_id_list, process_feature = self.extract_from_set(pcap_set)
            process_feature_list.append(process_feature)
            flow_id_list_all.append(flow_id_list)
        return flow_id_list_all, process_feature_list

    def extract_from_path(self, src_file_path):
        # 测试集中，已经有一个进程pcap，并不需要切分
        pcap_set = []
        with open(src_file_path, 'rb') as f:
            pcap = dpkt.pcap.Reader(f)
            for ts, pkt in pcap:
                pcap_set.append([ts, pkt])

        flow_id_list, process_feature = self.extract_from_set(pcap_set)
        return flow_id_list, process_feature

    def __sets2flow_table(self, pcap_set):
        flow_table = {}
        for ts, buf in pcap_set:
            try:
                eth = dpkt.ethernet.Ethernet(buf)
            except Exception as e:
                # print(e)
                continue
            if not isinstance(eth.data, dpkt.ip.IP):
                continue
            ip = eth.data
            tran = ip.data
            if isinstance(tran, dpkt.udp.UDP):
                protocol_tran = 'UDP'
            elif isinstance(tran, dpkt.tcp.TCP):
                protocol_tran = 'TCP'
            else:
                continue

            src_ip = socket.inet_ntoa(ip.src)  # socket.inet_ntoa(ip.src)
            dst_ip = socket.inet_ntoa(ip.dst)  # socket.inet_ntoa(ip.dst)

            flow_id = rf'{src_ip}:{tran.sport}-{protocol_tran}-{dst_ip}:{tran.dport}'
            flow_table.setdefault(flow_id, []).append([ts, buf])
        return flow_table

    def extract_from_set(self, pcap_set):

        process_feature = []
        flow_table = self.__sets2flow_table(pcap_set)
        flow_id_list = []
        for flow_id, flow in flow_table.items():
            sequence = []
            start_time = None
            ts_last = None
            for ts, buf in flow:
                start_time = ts if start_time is None else start_time
                ts_last = ts if ts_last is None else ts_last
                ip = dpkt.ethernet.Ethernet(buf).data
                tran = dpkt.ethernet.Ethernet(buf).data.data

                app_data = list(tran.data)[:20] if len(list(tran.data)) >= 20 else list(tran.data)[:20] + [0] * (20 - len(list(tran.data)))
                sequence.append([ip.len, ip.ttl, ip.p, ip.tos, tran.dport] + app_data)
                ts_last = ts
            flow_id_list.append(flow_id)
            process_feature.append(sequence)
        return flow_id_list, process_feature

    def save_process_feature_list(self, flow_id_list, process_feature_list, dst_dir_path):
        for i, process_feature in enumerate(process_feature_list):
            if len(process_feature) <= 0:
                continue
            with open(os.path.join(dst_dir_path, f'net_behavior_{uuid.uuid4()}.pkl'), 'wb') as f:
                pickle.dump([flow_id_list[i], process_feature], f)

    def merge_data(self, src_data_dir_path, dst_merged_data_dir_path, flag, label2name):
        data_x, data_y = [], []
        flow_id_list_all = []
        for i, category in label2name.items():
            category_path = os.path.join(src_data_dir_path, category)
            for root, dirs, files in os.walk(category_path):
                for file in files:
                    if not file.endswith('pkl'):
                        continue
                    src_file_path = os.path.join(root, file)
                    with open(src_file_path, 'rb') as f:
                        [flow_id_list, process_feature] = pickle.load(f)
                    flow_id_list_all += flow_id_list
                    data_x.append(process_feature)
                    data_y.append(i)

        data_y = torch.tensor(data_y)
        print(len(data_x), data_y.shape)

        with open(f'{dst_merged_data_dir_path}/id_{flag}.pkl', 'wb') as f:
            pickle.dump(flow_id_list_all, f)

        with open(f'{dst_merged_data_dir_path}/x_{flag}.pkl', 'wb') as f:
            pickle.dump(data_x, f)

        with open(f'{dst_merged_data_dir_path}/y_{flag}.pkl', 'wb') as f:
            pickle.dump(data_y, f)

        with open(os.path.join(dst_merged_data_dir_path, 'N.txt'), 'w') as f:
            f.write('')


    def run(self, src_dataset_dir_path, tmp_data_root_dir_path, dst_merged_data_dir_path, flag, label2name):
        # src_dataset_dir_path = r"F:\EN2025_both"
        # tmp_data_root_dir_path = r'./data/EN2025_both'
        # dst_merged_data_dir_path = '../data/EN2025_both'

        for root, dirs, files in os.walk(src_dataset_dir_path):
            for file in files:
                if not file.endswith('pcap'):
                    continue

                src_file_path = os.path.join(root, file)

                if os.path.getsize(src_file_path) <= 24:
                    continue

                print(src_file_path)
                dst_rel_path = os.path.relpath(os.path.dirname(src_file_path), src_dataset_dir_path)
                dst_data_dir_path = os.path.join(tmp_data_root_dir_path, dst_rel_path)

                # sequence_tensor = b.extract_from_file(src_file_path)
                # b.save_sequence_tensor(sequence_tensor, dst_dir_path)

                flow_id_list, process_feature_list = self.split_and_extract(src_file_path, './tmp')
                if not os.path.exists(dst_data_dir_path):
                    os.makedirs(dst_data_dir_path)
                # flow_id_list 为一个进程的所有流
                self.save_process_feature_list(flow_id_list, process_feature_list, dst_data_dir_path)

        self.merge_data(tmp_data_root_dir_path, dst_merged_data_dir_path, flag, label2name)