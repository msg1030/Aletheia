import torch
from model import EncoderModel

@torch.no_grad()
def infer_forward(model, tensor, device):
    model.eval()
    tensor = tensor.unsqueeze(0).to(device)
    z = model(tensor)

    return z.squeeze(0)


def cosine_similarity(z1, z2):
    cos_sim = F.cosine_similarity(z1, z2, dim=-1)

    return cos_sim.mean().item()


