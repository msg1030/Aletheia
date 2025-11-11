from data import LoadDataset
from model import EncoderModel
from train import train

import os
from tqdm import tqdm
import torch
import numpy as np


#mode = 'img' 
mode = 'entropy'
patch_size = 64 
window_size = 15
num_bins = 128
emb_dim = 256 
epochs = 300
batch_size = 40000
num_workers = 0 
lr = 1e-4

if __name__ == "__main__":
    device = torch.device("cuda")
    dataset = LoadDataset(mode, device, patch_size, window_size, num_bins)
 
    model = EncoderModel(patch_size, emb_dim)

    print("start train")
    trained_model = train(model, dataset, device, epochs, batch_size, lr)
   
    os.makedirs("../checkpoints", exist_ok=True)
    model_path = "../checkpoints/model.pt"
    torch.save(trained_model.state_dict(), model_path)
    print("Model saved")

