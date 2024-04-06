import torch
from torch import nn


class GCNBlock(nn.Module):
    def __init__(self, in_channels, inter_channels):
        super(GCNBlock, self).__init__()

        self.g = nn.Conv1d(
            in_channels=in_channels,
            out_channels=inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.W = nn.Sequential(
            nn.Conv1d(
                in_channels=inter_channels,
                out_channels=in_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm1d(in_channels),
        )
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)

        self.theta = nn.Conv1d(
            in_channels=in_channels,
            out_channels=inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.phi = nn.Conv1d(
            in_channels=in_channels,
            out_channels=inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, v):
        v = v.permute(0, 2, 1)

        g_v = self.g(v)
        g_v = g_v.permute(0, 2, 1)

        theta_v = self.theta(v)
        theta_v = theta_v.permute(0, 2, 1)
        phi_v = self.phi(v)
        R = torch.bmm(theta_v, phi_v)
        N = R.size(-1)
        R_normalized = R / N

        y = torch.bmm(R_normalized, g_v)
        y = y.permute(0, 2, 1)
        W_y = self.W(y)
        v_star = W_y + v
        v_star = v_star.permute(0, 2, 1)

        return v_star
