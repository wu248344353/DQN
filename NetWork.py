import torch.nn as nn
import torch


class NET(nn.Module):
    """定义神经网络类"""
    def __init__(self, state_dim, action_dim):
        super(NET, self).__init__()
        self.lin1 = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU()
        )
        self.lin2 = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU()
        )
        self.lin3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.lin4 = nn.Linear(256, action_dim)

    def forward(self, state):
        state = state.to(torch.float32)
        feature = self.lin1(state)
        feature = self.lin2(feature)
        feature = self.lin3(feature)
        out_feature = self.lin4(feature)
        return out_feature
