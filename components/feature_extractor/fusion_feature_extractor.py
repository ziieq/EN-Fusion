from .end_feature_extractor import EndBehaviorExtractor
from .net_feature_extractor import NetBehaviorExtractor
import os
import pickle
import tqdm



class FusionFeatureExtractor:
    def __init__(self):
        pass

    def extract_1_class(self, src_dir_path, end_extractor, net_extractor):
        x = {}
        hash_dict = {}
        for root, dirs, files in os.walk(src_dir_path):
            graph, sequence = None, None
            for file in files:

                # End-side
                if file.endswith('json'):
                    src_file_path = os.path.join(root, file)
                    graph = end_extractor.extract_file_to_data(src_file_path, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'api2intro.txt'))
                    if graph is None:
                        print('graph is None')
                        break

                # Net-side
                if file.endswith('pcap'):
                    src_file_path = os.path.join(root, file)
                    if os.path.getsize(src_file_path) <= 24:
                        print(rf'{root}: pcap is too small')

                    flow_id_list, sequence = net_extractor.extract_from_path(src_file_path)
                    if len(sequence) <= 0:
                        print(rf'{root}: pcap is too small')
                        sequence = None


            if graph is not None:
                print(rf'{root}: append')
                x[root] = [graph, sequence]
                hash_str = os.path.basename(root)
                hash_dict[root]= hash_str
        print(len(x.keys()))
        return x, hash_dict


    def extract_all_classes(self, src_dataset_dir_path, label2name):
        end_extractor = EndBehaviorExtractor()
        net_extractor = NetBehaviorExtractor()
        x_all, y_all = [], []
        hash_list = []
        for idx, category in label2name.items():
            category_path = os.path.join(src_dataset_dir_path, category)
            x, hash_dict = self.extract_1_class(category_path, end_extractor, net_extractor)
            for root, features in x.items():
                x_all.append(features)
                y_all.append(idx)
                hash_list.append(hash_dict[root])
            print(f'{category}: {len(list(x.keys()))}')
        return hash_list, x_all, y_all

    def save(self, data, data_path):
        with open(data_path, 'wb') as f:
            pickle.dump(data, f)

    def run(self, src_dataset_dir_path, dst_data_dir_path, flag, label2name):
        hash_list, x_all, y_all = self.extract_all_classes(src_dataset_dir_path, label2name)
        # x = [[end, net], [end, net], [end, net], ...]
        os.makedirs(dst_data_dir_path, exist_ok=True)
        self.save(hash_list, os.path.join(dst_data_dir_path, f'id_{flag}.pkl'))
        self.save(x_all, os.path.join(dst_data_dir_path, f'x_{flag}.pkl'))
        self.save(y_all, os.path.join(dst_data_dir_path, f'y_{flag}.pkl'))
        with open(os.path.join(dst_data_dir_path, 'EN.txt'), 'w') as f:
            f.write('')


# if __name__ == '__main__':
#
#     src_dataset_dir_path = r"F:\EN2025_full"
#     dst_data_dir_path = r'../data/fusion/EN2025_full'
#
#     x_all, y_all = extract_all_classes(src_dataset_dir_path)
#
#     # x = [[end, net], [end, net], [end, net], ...]
#     os.makedirs(dst_data_dir_path, exist_ok=True)
#     save(x_all, os.path.join(dst_data_dir_path, 'data_x.pkl'))
#     save(y_all, os.path.join(dst_data_dir_path, 'data_y.pkl'))
