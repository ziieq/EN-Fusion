import os
import pickle
from transformers import BertTokenizer
from transformers import BertModel, BertConfig, BertForSequenceClassification
import torch


class BertEncoder:

    def __init__(self, dict_name):
        root_path = os.path.dirname(os.path.abspath(__file__))
        bert_path = os.path.join(root_path, 'bert')
        self.dict_path = os.path.join(root_path, f'{dict_name}.pkl')

        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.bert_model = BertModel.from_pretrained(bert_path, config=BertConfig.from_pretrained(bert_path))

        self.word_dict = {}
        self.__read_word_dict()

    def encode_str(self, sentence):

        # 若预测单词已保存在字典中，直接返回
        if sentence in self.word_dict:
            return self.word_dict[sentence]

        # 开始预测单词
        input_ids = self.tokenizer(sentence, return_tensors='pt')
        with torch.no_grad():
            outputs = self.bert_model(**input_ids).last_hidden_state.mean(dim=1)

        # 预测的结果保存在字典中
        self.word_dict[sentence] = outputs
        return outputs

    def __read_word_dict(self):

        if not os.path.exists(self.dict_path):
            return
        with open(self.dict_path, 'rb') as f_read:
            self.word_dict = pickle.load(f_read)

    def save_word_dict(self):
        with open(self.dict_path, 'wb') as f_save:
            pickle.dump(self.word_dict, f_save)


if __name__ == '__main__':
    en = BertEncoder('test')
    res = en.encode_str('RegOpenKey with Weight 24')
    print(res)