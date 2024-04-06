import torch.nn as nn

from .caption_encoder import CaptionEncoder
from .caption_generator import CaptionGenerator
from .image_encoder import ImageEncoder


class VSRN(nn.Module):
    def __init__(self, config):
        super(VSRN, self).__init__()
        self.image_encoder = ImageEncoder(config.dim_image, config.dim_embed)
        self.caption_encoder = CaptionEncoder(
            config.vocab_size, config.dim_word, config.dim_embed
        )
        self.caption_generator = CaptionGenerator(
            config.dim_embed, config.dim_hidden, config.dim_word, config.vocab_size
        )

    def forward(self, images, captions, valid_length):
        img_feats, gcn_img_feats = self.image_encoder(images)
        caption_feats = self.caption_encoder(captions, valid_length)
        caption_generator_ouput = self.caption_generator(
            gcn_img_feats, captions[:, :-1]
        )
        return img_feats, caption_feats, caption_generator_ouput
