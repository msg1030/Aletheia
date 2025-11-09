import os
import rasterio
from rasterio.transform import Affine
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np

def read_img(main_dir, jgw_name, tif_name, crs='EPSG:4326'):
    jgw_path = os.path.join(main_dir, jgw_name)
    tif_path = os.path.join(main_dir, tif_name)

    with open(jgw_path, 'r') as f:
        vals = [float(line.strip()) for line in f.readlines()[:6]]

    A, D, B, E, C, F = vals
    transform = Affine(A, B, C, D, E, F)

    with rasterio.open(tif_path) as src:
        img = src.read(1)
        profile = src.profile

    profile.update({
        "transform": transform,
        "crs": crs
    })

    return img, profile

class K5:
    main_dir1 = '../data/2. InSAR pair 영상_K5_미국 샌프란시스코/K5_20220129135232_000010_46340_A_ES05_HH_SCS_B_L1A'
    tif_name1 = 'K5_20220129135232_000010_46340_A_ES05_HH_SCS_B_L1A_GIM.tif'
    jgw_name1 = 'K5_20220129135232_000010_46340_A_ES05_HH_SCS_B_L1A_br.jgw'

    main_dir2 = '../data/2. InSAR pair 영상_K5_미국 샌프란시스코/K5_20220226135243_000010_46761_A_ES05_HH_SCS_B_L1A'
    tif_name2 = 'K5_20220226135243_000010_46761_A_ES05_HH_SCS_B_L1A_GIM.tif'
    jgw_name2 = 'K5_20220226135243_000010_46761_A_ES05_HH_SCS_B_L1A_br.jgw'

    def load_img():
        imgs = [] 
        (img, profile) = read_img(K5.main_dir1, K5.jgw_name1, K5.r_tif_name1); imgs.append(img)
        (img, profile) = read_img(K5.main_dir2, K5.jgw_name2, K5.r_tif_name2); imgs.append(img)

        return imgs


class K3:
    main_dir1 = '../data/1. 광학 입체영상_K3_미국 샌프란시스코/K3_20220123211603_51687_16811282_L1R'
    b_tif_name1 = 'K3_20220123211603_51687_16811282_L1R_B.tif'
    g_tif_name1 = 'K3_20220123211603_51687_16811282_L1R_G.tif'
    n_tif_name1 = 'K3_20220123211603_51687_16811282_L1R_N.tif'
    p_tif_name1 = 'K3_20220123211603_51687_16811282_L1R_P.tif'
    r_tif_name1 = 'K3_20220123211603_51687_16811282_L1R_R.tif'
    jgw_name1 = 'K3_20220123211603_51687_16811282_L1R_br.jgw'

    main_dir2 = '../data/1. 광학 입체영상_K3_미국 샌프란시스코/K3_20220123211718_51687_16811282_L1R'
    b_tif_name2 = 'K3_20220123211718_51687_16811282_L1R_B.tif'
    g_tif_name2 = 'K3_20220123211718_51687_16811282_L1R_G.tif'
    n_tif_name2 = 'K3_20220123211718_51687_16811282_L1R_N.tif'
    p_tif_name2 = 'K3_20220123211718_51687_16811282_L1R_P.tif'
    r_tif_name2 = 'K3_20220123211718_51687_16811282_L1R_R.tif'
    jgw_name2 = 'K3_20220123211718_51687_16811282_L1R_br.jgw'

    def load_img():
        imgs = []

        (img, profile) = read_img(K3.main_dir1, K3.jgw_name1, K3.b_tif_name1); imgs.append(img)
        (img, profile) = read_img(K3.main_dir1, K3.jgw_name1, K3.g_tif_name1); imgs.append(img)
        (img, profile) = read_img(K3.main_dir1, K3.jgw_name1, K3.n_tif_name1); imgs.append(img)
        (img, profile) = read_img(K3.main_dir1, K3.jgw_name1, K3.p_tif_name1); imgs.append(img)
        (img, profile) = read_img(K3.main_dir1, K3.jgw_name1, K3.r_tif_name1); imgs.append(img)

        (img, profile) = read_img(K3.main_dir2, K3.jgw_name2, K3.b_tif_name2); imgs.append(img)
        (img, profile) = read_img(K3.main_dir2, K3.jgw_name2, K3.g_tif_name2); imgs.append(img)
        (img, profile) = read_img(K3.main_dir2, K3.jgw_name2, K3.n_tif_name2); imgs.append(img)
        (img, profile) = read_img(K3.main_dir2, K3.jgw_name2, K3.p_tif_name2); imgs.append(img)
        (img, profile) = read_img(K3.main_dir2, K3.jgw_name2, K3.r_tif_name2); imgs.append(img)

        return imgs


class LoadDataset(Dataset):
    def __init__(self, mode='img', device='cuda', patch_size: int=64, window_size=9, num_bins=32):
        self.patches = []
        
        imgs = K3.load_img()

        for img in imgs:
            img = img[np.newaxis, ...].astype(np.float32)
            img = self._crop_nonzero_region(img)
            if mode == 'entropy':
                img = self._cal_entropy(img, device, window_size, num_bins)
            patches = self._split_into_patches(img, patch_size)
            self.patches.append(patches)
        self.patches = np.concatenate(self.patches, axis=0)  # [patch_count, patch_size, patch_size]
        self.patches = torch.tensor(self.patches, device='cuda') # cpu to gpu
        if mode == 'img':
            self.patches = self.patches.float() / self.patches.max()
            
    def _crop_nonzero_region(self, img):
        mask = img.sum(axis=0)
        rows = np.any(mask != 0 , axis=1)
        cols = np.any(mask != 0, axis=0)
    
        if not rows.any() or not cols.any():
            return img
    
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
    
        cropped = img[:, rmin:rmax+1, cmin:cmax+1]
        return cropped

    def _split_into_patches(self, img, patch_size):
        _, H, W = img.shape

        H_cropped = H - (H % patch_size)
        W_cropped = W - (W % patch_size)
        img_cropped = img[:, :H_cropped, :W_cropped]

        patches = []
        for i in range(0, H_cropped, patch_size):
            for j in range(0, W_cropped, patch_size):
                patch = img_cropped[:, i:i+patch_size, j:j+patch_size]
                patches.append(patch)

        patches = np.stack(patches, axis=0)  # [N, C, patch_size, patch_size]
    
        if patches.shape[1] == 1:
            patches = patches[:, 0, :, :]  # [N, patch_size, patch_size]

        return patches

    def _cal_entropy(self, img: np.ndarray, device, window_size=9, num_bins=32):
        # input shape : [1, H, W]
        tensor = torch.from_numpy(img).float().to(device)
        tensor = tensor / tensor.max()  # 0~1 normalization

        pad = window_size // 2
        tensor = tensor.unsqueeze(0)  # [1,1,H,W]

        unfolded = F.unfold(tensor, kernel_size=window_size, padding=pad)
        local_patches = unfolded.squeeze(0).T.contiguous()  # [H*W, K*K]
    
        bins = torch.linspace(0, 1, num_bins + 1, device=tensor.device)
        hist = torch.bucketize(local_patches, bins) - 1
        hist = hist.clamp(min=0, max=num_bins-1)

        H = torch.zeros(local_patches.shape[0], num_bins, device=tensor.device)
        H.scatter_add_(1, hist, torch.ones_like(hist, dtype=torch.float32))
        H /= H.sum(dim=1, keepdim=True)
    
        # -Σ p log2(p)
        entropy = -(H * torch.log2(H + 1e-9)).sum(dim=1)
        entropy = entropy.view(tensor.shape[2], tensor.shape[3])
    
        return entropy.cpu().numpy()

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]
        return torch.tensor(patch)  # [C, patch, patch]
