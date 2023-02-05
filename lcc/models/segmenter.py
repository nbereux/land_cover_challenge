from transformers import ViTConfig, ViTModel
import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingPatch(nn.Module):

    def __init__(self, in_size, patch_size, embed_dim=768):
        super().__init__()
        # Assert image size and patch size are divisible by each other
        C, H, W = in_size
        assert (H % patch_size == 0) and (W % patch_size == 0), "Image dimensions must be divisible by the patch size."
        self.patch_size = patch_size
        self.num_patches = (H // patch_size) * (W // patch_size)
        self.patch_embedding = nn.Conv2d(C, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        #print(x.shape)
        x = self.patch_embedding(x)
        #print(x.shape)
        x = x.flatten(2).transpose(1, 2)
        return x

class Encoder(nn.Module):

    def __init__(self, in_size, patch_size, embed_dim=768, depth=12, num_heads=12, mlp_dim=3072):
        super().__init__()
        self.patch_embed = EmbeddingPatch(in_size, patch_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.patch_embed.num_patches + 1, embed_dim))
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=mlp_dim)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=depth)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, :(x.shape[1]), :]
        #print(x.shape)
        x = self.transformer(x)
        return x

class DecoderLinear(nn.Module):
    
    def __init__(self, in_size, n_classes):
        super().__init__()
        self.linear = nn.Linear(in_size, n_classes)

    def forward(self, x, H, W, patch_size):
        x = self.linear(x) # shape is batch_size, n_patch, n_classes
        #print(x.shape)
        x = x.reshape(x.shape[0], int(H/patch_size), int(W/patch_size), x.shape[2]) # transform to batch_size, height/patch_size, width/patch_size, n_classes
        return x.permute(0, 3, 1, 2) # transform to batch_size, n_classes, height/patch_size, width/patch_size

class Segmenter(nn.Module):

    def __init__(self, in_size, n_classes, patch_size, embed_dim=768, depth=12, num_heads=12, mlp_dim=3072):
        super().__init__()
        self.encoder = Encoder(in_size, patch_size, embed_dim, depth, num_heads, mlp_dim)
        self.decoder = DecoderLinear(embed_dim, n_classes)
        self.name = 'Segmenter'

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.encoder(x)
        x = self.decoder(x, H, W, self.encoder.patch_embed.patch_size)
        #print(x.shape)
        x = F.interpolate(x, size=(H, W), mode='bilinear')
        #print(x.shape)
        return x