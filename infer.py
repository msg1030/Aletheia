import torch
from model import EncoderModel
from data import VaildStep 

@torch.no_grad()
def infer_forward(model, tensor, device):
    model.eval()
    tensor = tensor.unsqueeze(0).to(device)
    z = model(tensor)

    return z.squeeze(0)


def cosine_similarity(z1, z2):
    cos_sim = F.cosine_similarity(z1, z2, dim=-1)

    return cos_sim.mean().item()


#mode = 'img'
mode = 'entropy'
model_path = "../checkpoints/entropy_300.pt"

device = torch.device("cuda")
model = EncoderModel(patch_size=64, emb_dim=64)
model.load_state_dict(torch.load(model_path, map_location=device))

model.to(device)
model.eval() 

target_patches, vaild_patches = VaildStep.valid_patchs
