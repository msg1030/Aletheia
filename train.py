import torch
import torchvision.transforms.functional as TF
from model import EncoderModel, contrastive_loss

def augment_patch(patch):
    if torch.rand(1, device=patch.device) < 0.5:
        patch = torch.flip(patch, dims=[2])  # horizontal flip
    if torch.rand(1, device=patch.device) < 0.5:
        patch = torch.flip(patch, dims=[1])  # vertical flip
    
    if torch.rand(1, device=patch.device) < 0.5:
        angle = torch.empty(1).uniform_(0, 360).item()
        patch = TF.rotate(patch, angle, interpolation=TF.InterpolationMode.BILINEAR)

    patch = patch + 0.01 * torch.randn_like(patch)

    return patch

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
