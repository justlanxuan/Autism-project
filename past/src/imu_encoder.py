import torch
import torch.nn as nn
import torch.nn.functional as F


class IMUEncoder(nn.Module):

    def __init__(self, input_dim=9, hidden_dim=128, embed_dim=256):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.gru = nn.GRU(128, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, embed_dim)

    def forward(self, x):
        # x:[B,T,9]
        x = x.transpose(1, 2)       # -> [B,9,T]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.transpose(1, 2)       # -> [B,T,128]
        _, h = self.gru(x)          # h:[2,B,H]
        h = torch.cat([h[0], h[1]], dim=-1)
        emb = F.normalize(self.fc(h), dim=-1)
        return emb