"""
feature_extraction.py

This module performs a two-stage feature extraction process using the official
Mahmood Lab models from Hugging Face.
"""
import os
import glob
import openslide
import pandas as pd
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import AutoModel

from source.config import locations
from source.config import config

# This will execute once when the script starts.
# It relies on a token being stored locally after the first login.
# To log in for the first time, you can run: huggingface-cli login
print("--- Authenticating with Hugging Face Hub ---")

class PatchDataset(Dataset):
    def __init__(self, wsi, coords, patch_size_level0, transform):
        self.wsi = wsi
        self.coords = coords
        self.patch_size_level0 = (patch_size_level0, patch_size_level0)
        self.transform = transform

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        x, y = self.coords[idx]
        patch_pil = self.wsi.read_region((x, y), 0, self.patch_size_level0).convert("RGB")
        return self.transform(patch_pil)

# --- Stage 1: Patch-Level Feature Extraction using CONCH ---
def extract_patch_features(wsi_path, coords_csv_path, output_h5_path, model, transform, device, params):
    print(f"  Stage 1: Extracting patch features for {os.path.basename(wsi_path)}")
    try:
        wsi = openslide.OpenSlide(wsi_path)
        coordinates = pd.read_csv(coords_csv_path)[['x', 'y']].values
    except Exception as e:
        print(f"  Error opening files: {e}")
        return False

    try:
        base_mag = float(wsi.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
    except (KeyError, ValueError):
        base_mag = 40.0

    patch_size_read_level0 = int(params["feat_patch_size"] * (base_mag / config.TILE_MAGNIFICATION))
    dataset = PatchDataset(wsi, coordinates, patch_size_read_level0, transform)
    dataloader = DataLoader(dataset, batch_size=params["feat_batch_size"], num_workers=4)

    all_features = []
    model.eval()
    with torch.no_grad():
        for i, patch_batch in enumerate(dataloader):
            print(f"\r    Processing patch batch {i+1}/{len(dataloader)}", end="")
            patch_batch = patch_batch.to(device, dtype=torch.float16 if device == 'cuda' else torch.float32)
            features = model(patch_batch)
            all_features.append(features.cpu().numpy())
    print()

    features_np = np.vstack(all_features)
    try:
        with h5py.File(output_h5_path, 'w') as hf:
            hf.create_dataset('features', data=features_np)
            hf.create_dataset('coords', data=coordinates)
        print(f"  Patch features saved to: {output_h5_path}")
        wsi.close()
        return True
    except Exception as e:
        print(f"  Error saving H5 file: {e}")
        wsi.close()
        return False

# --- Stage 2: Slide-Level Feature Extraction using TITAN ---
def extract_slide_feature(patch_h5_path, output_npy_path, model, device):
    print(f"  Stage 2: Aggregating features with TITAN for {os.path.basename(patch_h5_path)}")
    try:
        with h5py.File(patch_h5_path, 'r') as hf:
            patch_features = torch.from_numpy(hf['features'][:])
    except Exception as e:
        print(f"  Error reading H5 file: {e}")
        return

    model.eval()
    with torch.no_grad():
        patch_features = patch_features.unsqueeze(0).to(device)
        slide_feature = model.forward_slide_features(patch_features)
        slide_feature_np = slide_feature.cpu().numpy()

    try:
        np.save(output_npy_path, slide_feature_np)
        print(f"  Slide feature saved to: {output_npy_path}")
    except Exception as e:
        print(f"  Error saving slide feature: {e}")


def process_all_slides(max_slides=None):
    # --- Load Official Models ---
    print("--- Loading official Mahmood Lab models from Hugging Face ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        titan_model = AutoModel.from_pretrained('MahmoodLab/TITAN', trust_remote_code=True)
        conch_model, conch_transform = titan_model.return_conch()

        titan_model = titan_model.to(device)
        conch_model = conch_model.to(device)

        if device == 'cuda':
            titan_model.half()
            conch_model.half()
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Fatal Error: Could not load models. {e}")
        return

    params = {
        "feat_patch_size": config.FEAT_PATCH_SIZE,
        "feat_batch_size": config.FEAT_BATCH_SIZE,
    }

    wsi_dir = locations.get_dataset_dir()
    segmentation_base_dir = locations.get_segmentation_output_dir()
    segmentation_dirs = sorted([d for d in glob.glob(os.path.join(segmentation_base_dir, "*")) if os.path.isdir(d)])

    if max_slides is not None:
        segmentation_dirs = segmentation_dirs[:max_slides]

    for slide_dir in segmentation_dirs:
        slide_name = os.path.basename(slide_dir)
        print(f"\n--- Processing: {slide_name} ---")

        wsi_path = os.path.join(wsi_dir, f"{slide_name}.tif")
        coords_csv_path = os.path.join(slide_dir, "coordinates.csv")
        patch_h5_path = os.path.join(slide_dir, "features_conch_v15.h5")
        slide_npy_path = os.path.join(slide_dir, "feature_slide_titan.npy")

        if not os.path.exists(wsi_path) or not os.path.exists(coords_csv_path):
            print("  Skipping: Missing WSI or coordinates.csv file.")
            continue

        if os.path.exists(patch_h5_path):
            print(f"  Skipping patch extraction: {os.path.basename(patch_h5_path)} already exists.")
            success = True
        else:
            success = extract_patch_features(wsi_path, coords_csv_path, patch_h5_path, conch_model, conch_transform, device, params)

        if success:
            if os.path.exists(slide_npy_path):
                print(f"  Skipping slide aggregation: {os.path.basename(slide_npy_path)} already exists.")
            else:
                extract_slide_feature(patch_h5_path, slide_npy_path, titan_model, device)

    print("\n--- All slides have been processed. ---")


if __name__ == '__main__':
    print("--- Running 2-stage feature extraction script in standalone mode ---")
    process_all_slides(max_slides=2)
    print("\n--- Standalone script execution finished. ---")