Run ENFusion.py for training and evaluation. 

The dataset path and hyperparameters in the __main__ area need to be manually modified.
The parameters such as lr and batch for training need to be modified in ENFusion-open/components/dl/train_test_predict.py and ENFusion-open\components\dl\train_test_predict_flow_level.py.

Preparations needed before running:
1. Please store the model files of bert-base in the bert directory under ENFusion-open\components\feature_extractor\APIEncoder\lib\bert_encoder\bert.
2. Please configure the api_key of deepseek in ENFusion-open\components\feature_extractor\APIEncoder\deepseek\deepseek_requester.py, or replace it with other large language models.

The structures of the net-side, end-side, and fusion datasets are as follows:
<pre>
net_side_dataset
├─category1
│  ├─ 1.pcap
│  ├─ 2.pcap
│  └─ n.pcap
└─category2
    ├─ 1.pcap
    ├─ 2.pcap
    └─ n.pcap
</pre>

<pre>
end_side_dataset
├─category1
│  ├─ 1.json
│  ├─ 2.json
│  └─ n.json
└─category2
    ├─ 1.json
    ├─ 2.json
    └─ n.json
</pre>

<pre>
fusion_dataset
├─category1
│  ├─ malware1
│  │   ├─dump.pcap
│  │   └─report.json
│  ├─ malware2
│  │   ├─dump.pcap
│  │   └─report.json
│  └─ malware3
│       ├─dump.pcap
│       └─report.json
└─category2
    ├─ malware1
    │   ├─dump.pcap
    │   └─report.json
    ├─ malware2
    │   ├─dump.pcap
    │   └─report.json
    └─ malware3
         ├─dump.pcap
         └─report.json
</pre>