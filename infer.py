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

mode = 'img'
#mode = 'entropy'
model_path = "../checkpoints/img_300.pt"
patch_size = 64 
emb_dim = 256
window_size = 15
num_bins = 128

device = torch.device("cuda")
model = EncoderModel(patch_size, emb_dim)
model.load_state_dict(torch.load(model_path, map_location=device))

model.to(device)
model.eval() 

target_patches, vaild_patches = VaildStep.valid_patchs(mode, device, patch_size, window_size, num_bins)

# --- DEBUG BLOCK: 데이터 동일성 / 통계 체크
print("===== PATCH DATA CHECK =====")
print("target_patches.shape:", target_patches.shape)   # [N, H, W] 혹은 [N, C, H, W]
print("vaild_patches.shape: ", vaild_patches.shape)
# 비교: 완전 동일한지 (정확한 동일/근사)
same_exact = torch.equal(target_patches, vaild_patches)
same_close = torch.allclose(target_patches.float(), vaild_patches.float(), atol=1e-6)
print("Exact equal:", same_exact, "Allclose:", same_close)

# 한 패치의 통계
for i in [0, 1, max(0, target_patches.shape[0]-1)]:
    p = target_patches[i].float()
    q = vaild_patches[i].float()
    print(f"patch {i} stats: target min/mean/max = {p.min().item():.4g}/{p.mean().item():.4g}/{p.max().item():.4g}")
    print(f"patch {i} stats: valid  min/mean/max = {q.min().item():.4g}/{q.mean().item():.4g}/{q.max().item():.4g}")
    print(f"L2 diff (target vs valid) = {torch.norm(p - q).item():.6f}")

# --- MODEL/EMB DEBUG: 같은 입력을 두 번 넣어 동일한 임베딩 나오는지
print("\n===== EMBEDDING CHECK =====")
sample = target_patches[0].detach().clone()
# try both channel options: no-channel vs add-channel
def try_infer(tensor, desc):
    with torch.no_grad():
        t = tensor.unsqueeze(0).to(device)            # [1,H,W] or [1,C,H,W] depending
        emb = model(t).squeeze(0)
    return emb

emb1 = try_infer(sample, "as-is")
emb2 = try_infer(sample, "as-is")
print("emb shape:", emb1.shape)
print("emb equal exact:", torch.allclose(emb1, emb2, atol=1e-6))
print("emb norms:", emb1.norm().item(), emb2.norm().item())
print("any nan in emb:", torch.isnan(emb1).any().item())

# If model expects channel dim, try adding channel:
try:
    emb_c1 = model(sample.unsqueeze(0).unsqueeze(0).to(device)).squeeze(0)
    print("With added channel: emb norm", emb_c1.norm().item(), "any nan:", torch.isnan(emb_c1).any().item())
except Exception as e:
    print("Adding channel caused error (maybe model signature differs):", e)

# Check norms across dataset (small sample)
def batch_embs(patches, n=50):
    n = min(n, len(patches))
    embs = []
    for i in range(n):
        e = infer_forward(model, patches[i], device)
        embs.append(e)
    embs = torch.stack(embs)
    norms = torch.norm(embs, dim=1)
    print(f"emb norms: mean {norms.mean().item():.4g}, std {norms.std().item():.4g}, min {norms.min().item():.4g}, max {norms.max().item():.4g}")
    print("any nan:", torch.isnan(embs).any().item())
batch_embs(target_patches, n=20)
#여기까지 검증코드

target_embs = [infer_forward(model, patch, device) for patch in target_patches]
vaild_embs  = [infer_forward(model, patch, device) for patch in vaild_patches]

target_embs = torch.stack(target_embs)  # [N, D]
vaild_embs  = torch.stack(vaild_embs)   # [M, D]

print(f"target_embs: {target_embs.shape}, vaild_embs: {vaild_embs.shape}")

similarity = torch.zeros(len(target_patches), len(vaild_patches))
for i, t_emb in enumerate(target_embs):
    similarity[i] = F.cosine_similarity(t_emb.unsqueeze(0), vaild_embs, dim=-1)

sim_map = similarity[0].cpu().numpy()

valid_rows = int(np.sqrt(len(vaild_patches)))
valid_cols = valid_rows
sim_map_2d = np.zeros((valid_rows, valid_cols), dtype=sim_map.dtype)
sim_map_2d.flat[:len(vaild_patches)] = sim_map

plt.figure(figsize=(6, 6))
plt.imshow(sim_map_2d, cmap="hot")
plt.colorbar(label="Cosine Similarity")
plt.title("Where target image likely appears in valid image")
plt.axis("off")
plt.tight_layout()
plt.savefig("similarity_map.png", dpi=300)
plt.close()
