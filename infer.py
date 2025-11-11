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

def cosine_similarity(target_embs, vaild_embs, batch_size=1000):
    similarities = []
    num_batches = (vaild_embs.size(0) + batch_size - 1) // batch_size

    with torch.no_grad():
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, vaild_embs.size(0))
            v_batch = vaild_embs[start:end]  # [B, D]

            sim = F.cosine_similarity(
                target_embs.unsqueeze(1),   # [N, 1, D]
                v_batch.unsqueeze(0),       # [1, B, D]
                dim=-1
            )  # [N, B]

            similarities.append(sim.cpu())
            torch.cuda.empty_cache()

            progress = (batch_idx + 1) / num_batches * 100
            print(f"\r[Processing] Batch {batch_idx + 1}/{num_batches} ({progress:.2f}%)", end="")

    print("\nSimilarity computation complete")
    similarities = torch.cat(similarities, dim=1)  # [N, M]
    
    return similarities

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

similarity = cosine_similarity(target_embs, vaild_embs, batch_size=1000)

sim_map = similarity.mean(dim=0).cpu().numpy()

N = sim_map.size
side = int(np.ceil(np.sqrt(N)))

sim_map_2d = np.zeros((side, side), dtype=sim_map.dtype)
sim_map_2d.flat[:N] = sim_map

plt.figure(figsize=(6, 6))
plt.imshow(sim_map_2d, cmap="hot")
plt.colorbar(label="Cosine Similarity")
plt.title("Where target image likely appears in valid image")
plt.axis("off")
plt.tight_layout()
plt.savefig("similarity_map.png", dpi=300)
plt.close()
