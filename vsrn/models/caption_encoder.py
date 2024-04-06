import torch
import torch.nn as nn


def l2norm(X):
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


class CaptionEncoder(nn.Module):
    def __init__(self, vocab_size, dim_word, dim_embed):
        super(CaptionEncoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, dim_word)
        self.rnn = nn.GRU(dim_word, dim_embed, batch_first=True)

    def forward(self, x, valid_length):
        x = self.embed(x)
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(
            x, valid_length, batch_first=True
        )
        _, hidden_state = self.rnn(packed_input)
        output = hidden_state[-1]
        output = l2norm(output)
        return output
