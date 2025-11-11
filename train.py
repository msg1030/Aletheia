import torch
import torch.nn.functional as F
from model import EncoderModel, contrastive_loss

def augment_patch(batch):
    # batch: (B, H, W)
    B, H, W = batch.shape
    patches = batch.unsqueeze(1).clone()

        # horizontal flip
    mask = torch.rand(B, device=batch.device) < 0.5
    patches[mask] = torch.flip(patches[mask], dims=[3]) 

    # vertical flip
    mask = torch.rand(B, device=batch.device) < 0.5
    patches[mask] = torch.flip(patches[mask], dims=[2])

    # random rotation
    mask = torch.rand(B, device=batch.device) < 0.5
    if mask.any():
        angles = (torch.rand(mask.sum(), device=batch.device) * 360) * torch.pi / 180
        theta = torch.zeros(mask.sum(), 2, 3, device=batch.device)
        theta[:, 0, 0] = torch.cos(angles)
        theta[:, 0, 1] = -torch.sin(angles)
        theta[:, 1, 0] = torch.sin(angles)
        theta[:, 1, 1] = torch.cos(angles)
        grid = F.affine_grid(theta, patches[mask].size(), align_corners=False)
        patches[mask] = F.grid_sample(patches[mask], grid, mode='bilinear', padding_mode='border', align_corners=False)

    # gaussian noise
    patches = patches + 0.01 * torch.randn_like(patches)

    return patches.squeeze(1) 

def step_train(model, batch, optimizer, device):
    model.train()
    patch1 = augment_patch(batch)
    patch2 = augment_patch(batch)

    z1 = model(patch1)
    z2 = model(patch2)

    z1 = torch.nn.functional.normalize(z1, dim=-1)
    z2 = torch.nn.functional.normalize(z2, dim=-1)

    loss = contrastive_loss(z1, z2)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def train(model, dataset, device, epochs=5, batch_size=2, lr=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    patches = dataset.patches
    num_samples = patches.shape[0]

    for epoch in range(epochs):
        total_loss = 0
        perm = torch.randperm(num_samples, device=device)

        for i in range(0, num_samples, batch_size):
            batch_idx = perm[i:i+batch_size]
            batch = patches[batch_idx].to(device)

            loss = step_train(model, batch, optimizer, device)
            total_loss += loss

        print(f"[Epoch {epoch+1}] Loss: {total_loss / (num_samples / batch_size):.4f}")
        
        log_file = "train_log.txt"
        with open(log_file, "a") as f:
            f.write(f"[Epoch {epoch+1}] Loss: {total_loss / (num_samples / batch_size):.4f}\n")

    return model
