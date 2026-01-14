import os
from torchvision import transforms
import pickle
from .api_processor import APIProcessor
import torch
from torch_geometric.data import Data
from collections import defaultdict
from .APIEncoder.api_encoder import APIEncoder
import tqdm


class APICallGraphConstructor:
    """API调用序列转无重复边有向图（边权重为频率归一化值）"""

    def __init__(self, api_encoder):
        self.api2idx = {}  # API名称→节点ID映射
        self.idx2api = {}  # 节点ID→API名称映射
        self.pair_count = defaultdict(int)  # 调用对频次统计：{(src,dst): count}

        self.api_encoder = api_encoder

    def _build_api_index(self, api_sequences):
        """为所有唯一API分配节点ID"""
        unique_apis = set()
        for seq in api_sequences:
            unique_apis.update(seq)
        # 按出现顺序分配ID（也可按字母序/自定义规则）
        for idx, api in enumerate(sorted(unique_apis)):
            self.api2idx[api] = idx
            self.idx2api[idx] = api

    def _count_api_pairs(self, api_sequences):
        """统计连续API调用对的频次（无重复统计）"""
        self.pair_count.clear()
        for seq in api_sequences:
            if len(seq) < 2:  # 跳过长度<2的无效序列
                continue
            # 遍历连续调用对
            for i in range(len(seq) - 1):
                src_api = seq[i]
                dst_api = seq[i + 1]
                self.pair_count[(src_api, dst_api)] += 1

    def _normalize_weights(self, target_range=(0, 1)):
        """Min-Max归一化调用对频次到目标区间"""
        if not self.pair_count:
            return {}
        # 提取频次值
        counts = list(self.pair_count.values())
        min_cnt = min(counts)
        max_cnt = max(counts)

        # 处理所有频次相同的情况（避免除以0）
        if max_cnt == min_cnt:
            return {pair: target_range[0] for pair in self.pair_count}

        # 归一化计算
        range_min, range_max = target_range
        scale = range_max - range_min
        normalized_weights = {}
        for (src, dst), cnt in self.pair_count.items():
            norm_val = range_min + scale * (cnt - min_cnt) / (max_cnt - min_cnt)
            normalized_weights[(src, dst)] = norm_val
        return normalized_weights

    def construct_graph(self, api_sequences, api2intro_path, target_range=(0, 1), ):
        """
        核心方法：构造有向图
        :param api_sequences: API调用序列列表，例：[["A","B","C"], ["A","C","B"]]
        :param target_range: 权重归一化目标区间，默认[0,1]
        :return: PyG Data对象（含edge_index/edge_weight/x）、节点映射字典
        """
        # 步骤1：构建API→节点ID映射
        self._build_api_index(api_sequences)
        # 步骤2：统计调用对频次
        self._count_api_pairs(api_sequences)
        # 步骤3：归一化频次为边权重
        norm_weights = self._normalize_weights(target_range)

        # 步骤4：构建无重复边的edge_index
        edge_index = []
        edge_weight = []
        for (src_api, dst_api), weight in norm_weights.items():
            src_idx = self.api2idx[src_api]
            dst_idx = self.api2idx[dst_api]
            edge_index.append([src_idx, dst_idx])
            edge_weight.append(weight)

        # 转换为PyG标准张量格式
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weight, dtype=torch.float32)

        # 可选：构建节点特征（这里用one-hot，也可替换为自定义特征）
        # num_nodes = len(self.api2idx)
        # x = torch.eye(num_nodes, dtype=torch.float32)  # one-hot特征


        # 构造节点特征
        # with open(api2intro_path, 'r', encoding='utf-8') as f:
        #     api2intro = eval(f.read())

        sorted_items = sorted(self.idx2api.items(), key=lambda x: x[0], reverse=False)
        x = []
        for idx, api in sorted_items:



            embedding = self.api_encoder.encode(api)
            x.append(embedding)

        x = torch.tensor(x, dtype=torch.float)

        # 构建PyG Data对象
        graph_data = Data(
            x=x,  # 节点特征（num_nodes × num_features）
            edge_index=edge_index,  # 有向边索引[2, num_edges]
            edge_weight=edge_weight  # 归一化边权重[num_edges]
        )

        return graph_data, self.api2idx, self.idx2api

# 有向图，无重复边，有权重
class EndBehaviorExtractor:
    def __init__(self):
        self.toPIL = transforms.ToPILImage()
        self.p = APIProcessor()
        # self.ds = DeepSeek()
        root_path = os.path.dirname(os.path.abspath(__file__))
        api2intro = os.path.join(root_path, 'APIEncoder', 'api2intro.txt')
        self.api_encoder = APIEncoder(api2intro_path=api2intro, bert_pkl_path='api2intro', is_raw=False, dimention=128)


    def extract_file_to_data(self, src_file_path, api2intro_path):
        constructor = APICallGraphConstructor(self.api_encoder)
        api_list = self.read_api_from_file(src_file_path)
        graph_data, api2idx, idx2api = constructor.construct_graph(
            [api_list],
            api2intro_path,
            target_range=(0.1, 1),
        )
        return graph_data

    def read_api_from_file(self, src_file_path):
        api_list = self.p.extract_api_sequence(src_file_path, category_list=[], category_exclude_list=[])
        # (time_stamp, category, api, arguments)
        api_list = [api[2] for api in api_list]
        return api_list

    def image_tensor_to_PIL(self, image_tensor, save_path):
        pic = self.toPIL(image_tensor)
        pic.save(save_path)
        return pic

    def merge_all_data(self, src_dataset_dir_path, dst_dataset_dir_path, flag, label2name):
        data_x, data_y = [], []
        id_list = []
        for i, category in label2name.items():
            category_path = os.path.join(src_dataset_dir_path, category)
            for root, dirs, files in os.walk(category_path):
                for file in files:
                    if not file.endswith('pkl'):
                        continue
                    src_file_path = os.path.join(root, file)
                    print(f'\r{src_file_path}', end='')
                    with open(src_file_path, 'rb') as f_src:

                        g = pickle.load(f_src)
                        data_x.append(g)
                        data_y.append(i)
                        id_list.append(file.split('.')[0])

        with open(f'{dst_dataset_dir_path}/id_{flag}.pkl', 'wb') as f:
            pickle.dump(id_list, f)

        with open(f'{dst_dataset_dir_path}/x_{flag}.pkl', 'wb') as f:
            pickle.dump(data_x, f)

        with open(f'{dst_dataset_dir_path}/y_{flag}.pkl', 'wb') as f:
            pickle.dump(data_y, f)

        with open(os.path.join(dst_dataset_dir_path, 'E.txt'), 'w') as f:
            f.write('')

    def run(self, src_dataset_dir_path, data_embedding_dir_path, dst_data_root_dir_path, flag, label2name):
        api2intro = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'api2intro.txt')

        # 开始对数据集构图
        print('start walk...')
        for root, dirs, files in os.walk(src_dataset_dir_path):
            for file in tqdm.tqdm(files):
                if not file.endswith('json') and not file.endswith('xml'):
                    continue

                src_file_path = os.path.join(root, file)
                dst_rel_path = os.path.relpath(os.path.dirname(src_file_path), src_dataset_dir_path)
                dst_dir_path = os.path.join(data_embedding_dir_path, dst_rel_path)
                dst_file_path = os.path.join(dst_dir_path, f'{file}.pkl')
                if os.path.exists(dst_file_path):
                    print(f'\r jump:{file}', end='')
                    continue
                try:
                    g_data = self.extract_file_to_data(src_file_path, api2intro)
                except Exception as e:
                    print(e)
                    continue

                if g_data is None:
                    print('g_data is None')
                    continue
                if not os.path.exists(dst_dir_path):
                    os.makedirs(dst_dir_path)
                with open(dst_file_path, 'wb') as f:
                    pickle.dump(g_data, f)

        os.makedirs(dst_data_root_dir_path, exist_ok=True)
        self.merge_all_data(data_embedding_dir_path, dst_data_root_dir_path, flag, label2name)
