import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
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

target_patches, vaild_patches = VaildStep.valid_patchs()

target_embs = [infer_forward(model, patch, device) for patch in target_patches]
vaild_embs  = [infer_forward(model, patch, device) for patch in vaild_patches]

target_embs = torch.stack(target_embs)  # [N, D]
vaild_embs  = torch.stack(vaild_embs)   # [M, D]

print(f"target_embs: {target_embs.shape}, vaild_embs: {vaild_embs.shape}")

similarity = F.cosine_similarity(
    target_embs.unsqueeze(1),  # [N, 1, D]
    vaild_embs.unsqueeze(0),   # [1, M, D]
    dim=-1
)  # [N, M]

sim_map = similarity.mean(dim=0).cpu().numpy()

valid_rows = int(np.sqrt(vaild_patches.shape[0]))
valid_cols = valid_rows

sim_map_2d = sim_map.reshape(valid_rows, valid_cols)

plt.figure(figsize=(6, 6))
plt.imshow(sim_map_2d, cmap="hot")
plt.colorbar(label="Cosine Similarity")
plt.title("Where target image likely appears in valid image")
plt.axis("off")
plt.tight_layout()
plt.savefig("similarity_map.png", dpi=300)
plt.close()
