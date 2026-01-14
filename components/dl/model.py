import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
import torch_geometric.data as data


class DesnseDetector(nn.Module):
    def __init__(self, output_dim=2):
        super(DesnseDetector, self).__init__()

        self.dense_net = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim)
        )
        self.dense_end = torch.nn.Linear(128, output_dim)
        self.dense_fusion = nn.Sequential(
            nn.Linear(160, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim),
        )

    def forward(self, x, side):
        if side == 'end':
            return self.dense_end.forward(x)
        elif side == 'net':
            return self.dense_net.forward(x)
        elif side == 'fusion':
            return self.dense_fusion.forward(x)
        else:
            raise Exception('side must be end/net/fusion')


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.context = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden_states):
        attn_weights = torch.tanh(self.attn(hidden_states))
        attn_weights = self.context(attn_weights).squeeze(2)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context_vector = torch.sum(attn_weights.unsqueeze(2) * hidden_states, dim=1)
        return context_vector, attn_weights


class NetGRU(nn.Module):
    def __init__(self, middle_size=16):
        super(NetGRU, self).__init__()
        self.gru = nn.GRU(input_size=25, hidden_size=16, num_layers=1, batch_first=True, bidirectional=True)
        self.attention = Attention(32)
        self.attention_flow_spectrum = Attention(32)
        self.mlp = nn.Sequential(
            nn.Linear(32, middle_size),
        )
        self.gru_flow_spectrum = nn.GRU(input_size=middle_size, hidden_size=16, num_layers=1, batch_first=True, bidirectional=True)

    def forward_encoder(self, x):

        flow_spectrum = None
        for i in range(x.shape[0]):
            sample = x[i, :, :]
            out_new, hidden = self.gru(sample)
            context_vector, attn_weights = self.attention(out_new)
            out_new = self.mlp(context_vector)
            out_new = out_new.unsqueeze(0)
            flow_spectrum = out_new if flow_spectrum is None else torch.cat([flow_spectrum, out_new], dim=0)
        out, hidden = self.gru_flow_spectrum(flow_spectrum)
        context_vector, attn_weights = self.attention_flow_spectrum(out)

        return context_vector

class GATGraphClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GATGraphClassifier, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)

    def forward_encoder(self, data):
        x, edge_index = data.x, data.edge_index
        # 第一层GAT卷积
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # 第二层GAT卷积
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        # 全局平均池化
        x = global_mean_pool(x, data.batch)
        return x

class ENFusion(nn.Module):
    def __init__(self, output_dim=2):
        super(ENFusion, self).__init__()
        self.end_side = GATGraphClassifier(128, 128)
        self.net_side = NetGRU(16)
        self.dense_detector = DesnseDetector(output_dim)

    def forward(self, x, side):
        end_f, net_f = x[0], x[1]

        if side == 'end':
            out_end = self.end_side.forward_encoder(end_f)
            return self.dense_detector.forward(out_end, 'end')
        elif side == 'net':
            out_net = self.net_side.forward_encoder(net_f)
            return  self.dense_detector.forward(out_net, 'net')
        else:
            # 融合场景，去除单侧的参数梯度
            for para in self.end_side.parameters():
                para.requires_grad = False
            for para in self.net_side.parameters():
                para.requires_grad = False
            out_end = self.end_side.forward_encoder(end_f)
            out_net = self.net_side.forward_encoder(net_f)
            assert out_end.shape[0] == out_net.shape[0]
            x = torch.cat([out_end, out_net], dim=1)
            out = self.dense_detector.forward(x, 'fusion')
            return out


