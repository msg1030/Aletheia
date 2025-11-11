import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    def __init__(self, patch_size=16, emb_dim=128):
        super().__init__()
        self.proj = nn.Conv2d(1, emb_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = x.unsqueeze(1) # [B, H, W] -> [B, 1, H, W]
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        x = F.normalize(x, dim=-1)
        return x


class EntropyTransformer(nn.Module):
    def __init__(self, emb_dim=128, depth=4, heads=4, mlp_dim=256):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=heads, dim_feedforward=mlp_dim, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.norm(x.mean(dim=1))  # mean pooling
        return x


class EncoderModel(nn.Module):
    def __init__(self, patch_size=64, emb_dim=128):
        super().__init__()
        self.embed = PatchEmbed(patch_size, emb_dim)
        self.encoder = EntropyTransformer(emb_dim)
        self.proj_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, x):
        patches = self.embed(x)
        rep = self.encoder(patches)
        z = F.normalize(self.proj_head(rep), dim=-1)
        return z


def contrastive_loss(z1, z2, temperature=0.1):
    B, D = z1.shape
    sim = torch.mm(z1, z2.T) / temperature
    labels = torch.arange(B, device=z1.device)
    loss = F.cross_entropy(sim, labels)
    return loss

