import os.path
import json
import pickle
import torch
from transformers import BertTokenizer
from transformers import BertModel, BertConfig
from xml.dom.minidom import parse


class APIProcessor:
    def __init__(self):
        pass

    def __read_api_sequence_from_report_json(self, src_file_path):
        api_sequence = []  # [(time_stam, api, arguments), (time_stam, api, arguments), ...]
        category_dict = {}
        with open(src_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            report = json.loads(content)
            try:
                for api_call in report['behavior']['processes'][0]['calls']:
                    time_stamp = api_call['timestamp']
                    category = api_call['category']
                    api = api_call['api'].rstrip('Ex').rstrip('ExW').rstrip('ExA').rstrip('A').rstrip('W')
                    arguments = api_call['arguments']
                    api_sequence.append((time_stamp, category, api, arguments))
                    category_dict[category] = 1
            except Exception:
                return []
        # print(category_dict)
        return api_sequence

    def __read_api_sequence_from_report_xml(self, xml_file_path):
        # 读取文件
        dom = parse(xml_file_path)
        # 获取文档元素对象
        elem = dom.documentElement
        # 获取行为
        actions = elem.getElementsByTagName('action')
        action_list = []
        for action in actions:
            # 获取属性
            name = action.getAttribute('api_name').rstrip('Ex').rstrip('ExW').rstrip('ExA').rstrip('A').rstrip('W')
            if len(name) == 0:
                continue

            time_d = -1
            ret_value = action.getAttribute('ret_value')
            # 获取参数
            arg_list = action.getElementsByTagName('apiArg_list')[0]
            arg_v_list = []
            for arg in arg_list.getElementsByTagName('apiArg'):
                arg_value = arg.getAttribute('value')
                arg_v_list.append(arg_value)
            # 获取额外参数
            arg_list = action.getElementsByTagName('exInfo_list')[0]
            for arg in arg_list.getElementsByTagName('exInfo'):
                arg_value = arg.getAttribute('value')
                arg_v_list.append(arg_value)
                # (time_stamp, category, api, arguments)
                # {'name': name, 'time_d': time_d, 'ret_value': ret_value, 'arg_v_list': arg_v_list}
            action_list.append((time_d, -1, name, arg_v_list))
        return action_list[1:-1]

    def extract_api_sequence(self, src_report_file_path, category_list=None, category_exclude_list=None):
        # # (time_stamp, category, api, arguments)
        # 判断需求列表与排除列表是否激活
        is_category_list, is_exclude_list = False, False
        if category_list is not None and len(category_list) > 0:
            is_category_list = True
        if category_exclude_list is not None and len(category_exclude_list) > 0:
            is_exclude_list = True

        # 读取API
        if src_report_file_path.endswith('json'):
            api_sequence = self.__read_api_sequence_from_report_json(src_report_file_path)
        elif src_report_file_path.endswith('xml'):
            api_sequence = self.__read_api_sequence_from_report_xml(src_report_file_path)
        else:
            return None
        api_list = []
        for api in api_sequence:
            # (time_stamp, category, api, arguments)
            # 如果在排除列表里，跳过
            if is_exclude_list and api[1] in category_exclude_list:
                continue
            # 如果不在需求列表里，跳过
            if is_category_list and api[1] not in category_list:
                continue
            api_list.append(api)
        # print('序列长度：{}'.format(len(api_list)))
        return api_list

    def encode_api_sequence(self, encoder, api_name_list, max_len=-1):

        # bert encode unseen api
        api_name_set = set(api_name_list)
        for api_name in api_name_set:
            encoder.encode_str(api_name)
        encoder.save_word_dict()

        # encode api_name_list
        embedding_list = []
        for api_name in api_name_list:
            embedding = encoder.encode_str(api_name)
            embedding_list.append(embedding)
        if max_len != -1:
            embedding_list = embedding_list[:max_len]
        embedding_tensor = torch.cat(embedding_list, dim=0)
        return embedding_tensor


if __name__ == '__main__':
    api_processor = APIProcessor()
    res = api_processor.extract_api_sequence(r"F:\Dataset\open_dataset\Public_Avast_CTU_CAPEv2_Dataset_Full\public_full_reports\000a891f4bd2cb143cbf8f4764f510ecef3e0632c31fa83a9593bf9bf61f2e9b.json")
    print(res)
    # ('2024-12-29 22:44:44,991', 'system', 'NtClose', [{'name': 'Handle', 'value': '0x00000220'}])
