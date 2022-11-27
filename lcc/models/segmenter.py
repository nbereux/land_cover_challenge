from transformers import ViTConfig, ViTModel
import torch
import torch.nn as nn

class Segmenter(nn.Module):

    def __init__(self, encoder_config: ViTConfig, device: torch.device):
        super().__init__()
        self.encoder_config = encoder_config
        tmp_model = ViTModel(self.encoder_config)
        self.encoder_embedding = tmp_model.embeddings
        self.encoder = tmp_model.encoder
        self.decoder = DecoderLinear(257, encoder_config.patch_size, encoder_config.hidden_size)
        self.device = device

    def __call__(self, image: torch.Tensor):
        image = image.to(self.device)
        img_shape = (image.shape[-2], image.shape[-1])
        output = self.encoder_embedding(image)
        # print('in forward', output.shape)

        output = self.encoder(output)
        # print('in forward', output['last_hidden_state'].shape)
        output = self.decoder(output['last_hidden_state'], img_shape)
        return output

class DecoderLinear(torch.nn.Module):

    def __init__(self, n_tokens, patch_size, d_encoder):
        super().__init__()
        self.patch_size = patch_size
        self.linear = torch.nn.Linear(d_encoder, n_tokens)
    
    def forward(self, x, im_size):
        H, W = im_size
        GS = H // self.patch_size
        x = self.linear(x)
        return x[:, :H, :W]