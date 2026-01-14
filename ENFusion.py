import os
import time
import warnings
from sys import exception

warnings.filterwarnings("ignore")
from components.feature_extractor.end_feature_extractor import EndBehaviorExtractor
from components.feature_extractor.net_feature_extractor import NetBehaviorExtractor
from components.feature_extractor.fusion_feature_extractor import FusionFeatureExtractor
from components.dl.train_test_predict import run_training, split_dataset, write_output
from components.dl.train_test_predict_flow_level import run_training_flow_level
from components.metric_computer.metric_computer import compute_metrics
import uuid


class ENFusion:
    def __init__(self):
        pass

    def extract_feature_for_train_test(self, dataset_path, metadata_dir_path, tmp_dir_path, pkl_path, label2name):
        is_end, is_net, is_fusion = False, False, False
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith('.xml') or file.endswith('.json'):
                    is_end = True
                if file.endswith('.pcap'):
                    is_net = True
            if is_end and is_net:
                is_fusion = True
                is_end, is_net = False, False

        if is_end:
            end_feature_extractor = EndBehaviorExtractor()
            # 临时文件夹构建
            embedding_dir = os.path.join(tmp_dir_path, 'end_embedding' + str(time.time()))  # 各样本独立且label未数字化
            os.makedirs(embedding_dir)
            # 最终PKL文件夹
            dst_data_root_dir_path = os.path.join(tmp_dir_path, 'end_data_' + str(time.time()))  # 融合各样本且处理label，可直接用于训练
            os.makedirs(dst_data_root_dir_path)
            # 提取训练集和测试集特征
            dataset_test_path = dataset_path
            end_feature_extractor.run(dataset_test_path, embedding_dir, metadata_dir_path, 'all', label2name)

        elif is_net:
            net_feature_extractor = NetBehaviorExtractor()
            # 临时文件夹构建
            net_tmp_dir_test = os.path.join(tmp_dir_path, 'net_tmp_test' + str(time.time()))  # 各样本独立且label未数字化
            os.makedirs(net_tmp_dir_test)
            # 最终PKL文件夹
            dst_data_root_dir_path = os.path.join(tmp_dir_path, 'net_data_' + str(time.time()))  # 融合各样本且处理label，可直接用于训练
            os.makedirs(dst_data_root_dir_path)
            # 提取训练集和测试集特征
            dataset_test_path = dataset_path
            net_feature_extractor.run(dataset_test_path, net_tmp_dir_test, metadata_dir_path, 'all', label2name)

        elif is_fusion:
            fusion_feature_extractor = FusionFeatureExtractor()
            # 最终PKL文件夹
            dst_data_root_dir_path = os.path.join(tmp_dir_path, 'fusion_data_' + str(time.time()))  # 融合各样本且处理label，可直接用于训练
            os.makedirs(dst_data_root_dir_path)
            # 提取训练集和测试集特征
            fusion_feature_extractor.run(dataset_path, metadata_dir_path, 'all', label2name)

    def train_and_test(self, metadata_dir_path, dst_model_dir_path, epoch=20, f_len=200, s_len=20, output_dim=2, level='process'):

        # # 读取已有模型
        model = None

        # 进行训练
        file_list = os.listdir(metadata_dir_path)
        if 'E.txt' in file_list:
            side = 'end'
        elif 'N.txt' in file_list:
            side = 'net'
        elif 'EN.txt' in file_list:
            side = 'fusion'
        else:
            return

        print('*** Random split dataset')
        split_dataset(metadata_dir_path)

        if level == 'process':
            print('*** Process level training')
            id_list, y_true, y_pred, y_score = run_training(model, metadata_dir_path, side, output_dim, dst_model_dir_path, epoch, f_len, s_len)
        elif level == 'flow':
            print('*** Flow level training')
            id_list, y_true, y_pred, y_score = run_training_flow_level(model, metadata_dir_path, side, output_dim, dst_model_dir_path, epoch, f_len, s_len)
        else:
            raise exception('Unknown level')
        return id_list, y_true, y_pred, y_score


if __name__ == '__main__':

    side = 'end' # end, net or fusion. It needs to be determined according to the specified dataset.
    dataset_path = rf'F:\Dataset\open_dataset\Public_Avast_CTU_CAPEv2_Dataset_Full\avast_family'
    dataset_name = 'avast'  # dataset_name is used for naming the result file.
    level = 'process'  # process or flow. Level 'flow' can only be used by ned-side detection!
    epoch = 40
    output_dim = 10 # category count
    loops = 10
    f_len = 200 # hyperparameter k
    s_len = 20 # hyperparameter n

    tmp_dir_path = os.path.join('./tmp', str(uuid.uuid4()))
    metadata_dir_path = rf'./metadata/{side}/{dataset_name}'
    os.makedirs(tmp_dir_path)
    os.makedirs(metadata_dir_path, exist_ok=True)

    print('* Start to extract features for ' + dataset_name)
    class_list = os.listdir(dataset_path)
    label2name = {class_list.index(x): x for x in class_list}
    print(label2name)

    detector = ENFusion()
    if (not os.path.exists(metadata_dir_path)) or (len(os.listdir(metadata_dir_path)) == 0):
        detector.extract_feature_for_train_test(dataset_path, metadata_dir_path,
                                                tmp_dir_path, './pkl', label2name)

    print('* Start to train and evaluate model')
    res_file_path = os.path.join('result', rf'{dataset_name}_output{output_dim}_f_len{f_len}_s_len{s_len}_epoch{epoch}.txt')
    for loop in range(loops):
        print(f'** Start loop {loop+1}')
        id_list, y_true, y_pred, y_score = detector.train_and_test(metadata_dir_path, './pkl',
                                                          epoch=epoch, f_len=f_len, s_len=s_len, output_dim=output_dim, level=level)


        write_output(res_file_path, y_true, y_pred, y_score)

    compute_metrics(res_file_path)
