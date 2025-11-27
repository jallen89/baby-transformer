from src.layers import Encoder, Decoder
import torch.nn as nn

class Transformer(nn.Module):

    def __init__(self, encoder_vocab_size, decoder_vocab_size, model_size=256, stack_cnt=2):
        super(Transformer, self).__init__()

        self.encoder = Encoder(encoder_vocab_size, model_size=model_size, stack_cnt=stack_cnt)
        self.decoder = Decoder(decoder_vocab_size, model_size=model_size, stack_cnt=stack_cnt)

        self.output = nn.Linear(model_size, decoder_vocab_size)

    def forward(self, encoder_x, decoder_x):

        encoder_out = self.encoder(encoder_x)
        encoder_attention_mask = self.encoder.create_encoder_mask(encoder_x)
        decoder_out = self.decoder(decoder_x, encoder_out, encoder_out, encoder_attention_mask)

        o = self.output(decoder_out)

        return o