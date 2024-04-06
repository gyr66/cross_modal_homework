import torch
import torch.nn as nn


class LanguageModelLoss(nn.Module):
    def __init__(self):
        super(LanguageModelLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(reduction="none")

    def forward(self, output, target, mask):
        batch_size = output.shape[0]
        output = output.contiguous().view(-1, output.shape[-1])
        target = target.contiguous().view(-1)
        mask = mask.contiguous().view(-1)
        loss = self.loss(output, target)
        loss = torch.sum(loss * mask) / batch_size
        return loss


def cosine_sim(im, s):
    return im.mm(s.t())


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, im, s):
        scores = cosine_sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        cost_s = (self.margin + scores - d1).clamp(min=0)
        cost_im = (self.margin + scores - d2).clamp(min=0)

        mask = (torch.eye(scores.size(0)) > 0.5).to(im.device)
        cost_s = cost_s.masked_fill_(mask, 0)
        cost_im = cost_im.masked_fill_(mask, 0)

        # cost_s = cost_s.max(1)[0]
        # cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()
