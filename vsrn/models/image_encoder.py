import torch
import torch.nn as nn
from .gcn import GCNBlock


def l2norm(X):
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


class ImageEncoder(nn.Module):
    def __init__(self, dim_img, dim_embed):
        super(ImageEncoder, self).__init__()
        self.linear = nn.Linear(dim_img, dim_embed)
        self.rnn = nn.GRU(dim_embed, dim_embed, batch_first=True)
        self.gcn = nn.Sequential()

        for i in range(4):
            self.gcn.add_module(
                "GCN_{}".format(i),
                GCNBlock(in_channels=dim_embed, inter_channels=dim_embed),
            )
        self.bn = nn.BatchNorm1d(dim_embed)

    def forward(self, images):
        img_embed = self.linear(images)
        gcn_img_embed = self.gcn(img_embed)
        gcn_img_embed = l2norm(gcn_img_embed)
        _, hidden_state = self.rnn(gcn_img_embed)
        features = hidden_state[-1]
        features = self.bn(features)
        features = l2norm(features)
        return features, gcn_img_embed
