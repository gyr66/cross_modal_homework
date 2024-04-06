import torch
import torch.nn as nn

from .attention import Attention


class Encoder(nn.Module):
    def __init__(
        self,
        dim_input,
        dim_hidden,
        dropout=0.2,
    ):
        super(Encoder, self).__init__()
        self.dim_hidden = dim_hidden
        self.linear = nn.Linear(dim_input, dim_hidden)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(
            dim_hidden,
            dim_hidden,
            batch_first=True,
        )

    def forward(self, img_feats):
        img_feats = self.linear(img_feats)
        img_feats = self.dropout(img_feats)
        output, hidden_state = self.rnn(img_feats)
        return output, hidden_state


class Decoder(nn.Module):
    def __init__(
        self,
        dim_hidden,
        dim_word,
        vocab_size,
        dropout=0.1,
    ):
        super(Decoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, dim_word)
        self.attention = Attention(dim_hidden)
        self.rnn = nn.GRU(
            dim_hidden + dim_word,
            dim_hidden,
            batch_first=True,
        )
        self.out = nn.Linear(dim_hidden, vocab_size)

    def forward(self, encoder_output, encoder_hidden_state, input):
        decoder_hidden = encoder_hidden_state
        output = []
        input_emb = self.embedding(input)
        for i in range(input_emb.size(1)):
            current_words = input_emb[:, i, :]
            context = self.attention(decoder_hidden.squeeze(0), encoder_output)
            decoder_input = torch.cat([current_words, context], dim=1)
            decoder_input = self.dropout(decoder_input).unsqueeze(1)
            decoder_output, decoder_hidden = self.rnn(decoder_input, decoder_hidden)
            output.append(self.out(decoder_output))
        output = torch.cat(output, dim=1)
        return output


class CaptionGenerator(nn.Module):
    def __init__(self, dim_input, dim_hidden, dim_word, vocab_size):
        super(CaptionGenerator, self).__init__()
        self.encoder = Encoder(dim_input, dim_hidden)
        self.decoder = Decoder(dim_hidden, dim_word, vocab_size)

    def forward(self, encoder_input, decoder_input):
        encoder_output, encoder_hidden = self.encoder(encoder_input)
        decoder_output = self.decoder(encoder_output, encoder_hidden, decoder_input)
        return decoder_output
