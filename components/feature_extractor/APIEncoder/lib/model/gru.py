import torch
import torch.nn as nn


class GRU(nn.Module):
    def __init__(self):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=16, num_layers=2, batch_first=True, bidirectional=True)

    def forward(self, x):
        out, hidden = self.gru(x.to(torch.float32))
        x = out[:, -1, :]
        return x


if __name__ == '__main__':
    model = GRU()
    input = torch.rand(64, 6, 1)
    output = model(input)
    print(output.size())
