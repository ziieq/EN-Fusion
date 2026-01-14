import pandas as pd
import tqdm
from m1_end.fe.APIEncoder.lib.bert_encoder.bert_encoder import BertEncoder
import pickle
from deepseek_requester import DeepSeek




# def generate_bert_embedding_table_old():
#     bert = BertEncoder('api')
#     df = pd.read_csv('手动/api_deepseek.csv', header=None)
#     df.columns = ['api', 'content']
#     print(df.head())
#
#     bert_embedding_dict = {}
#     for index, row in df.iterrows():
#         print(f'\r encode: {row["api"]}', end='')
#         bert_embedding_dict[row['api']] = bert.encode_str(row['content'])[0]
#     bert.save_word_dict()
#     with open('./bert_embedding_dict.pkl', 'wb') as f:
#         pickle.dump(bert_embedding_dict, f)

def generate_bert_embedding_table(api2intro_path):
    bert = BertEncoder('api2intro_random.pkl')
    with open(api2intro_path, 'r', encoding='utf-8') as f:
        api2intro = eval(f.read())

    bert_embedding_dict = {}
    for key, content in api2intro.items():
        print(f'\r Encode: {key}', end='')
        bert_embedding_dict[key] = bert.encode_str(content)[0]
    bert.save_word_dict()
    with open('./bert_embedding_dict.pkl', 'wb') as f:
        pickle.dump(bert_embedding_dict, f)

def check_pkl():
    with open('./bert_embedding_dict.pkl', 'rb') as f:
        bert_embedding_dict = pickle.load(f)
        print(bert_embedding_dict)


def api_df2intro_dict(src_df_path, src_api2intro, dst_dict_path):
    def run_api2intro(api_list, api2intro):
        ds = DeepSeek()
        for api in tqdm.tqdm(api_list):
            if api in api2intro:
                continue
            content = ds.request(api)
            print(content)
            api2intro[api] = content
        return api2intro
    df = pd.read_csv(src_df_path, header=None)
    api_list = df.iloc[:, 0].to_list()

    with open(src_api2intro, 'r', encoding='utf-8') as f:
        api2intro = eval(f.read())

    api2intro = run_api2intro(api_list, api2intro)
    with open(dst_dict_path, 'w', encoding='utf-8') as f:
        f.write(str(api2intro))


if __name__ == '__main__':
    # api_df2intro_dict(r'E:\Main\Engineering\PyCharm\Project\IoT\P4_Fusion\feature_constructor\end_side\v3_graph\api_statistic_table\datacon2019_all.csv', './api2intro.txt', './api2intro.txt')
    generate_bert_embedding_table('./api2intro_random.txt')
    # check_pkl()
