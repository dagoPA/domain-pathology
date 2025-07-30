import os
import json
import numpy as np
import torch
import openslide

from source.config import locations
from .conchv1_5 import create_model_from_pretrained


def load_tile_coordinates(slide_dir, slide_name):
    npz_path = os.path.join(slide_dir, f"{slide_name}_tile_coords.npz")
    if os.path.exists(npz_path):
        data = np.load(npz_path)
        coords = data['coordinates']
        tile_size = int(data['tile_size'])
    else:
        json_path = os.path.join(slide_dir, f"{slide_name}_tile_coords.json")
        with open(json_path) as f:
            info = json.load(f)
        coords = np.array(info['coordinates'], dtype=np.int32)
        tile_size = info['tile_size']
    return coords, tile_size


def extract_features_for_slide(model, transform, wsi_path, coords, tile_size, batch_size=16, device='cpu'):
    slide = openslide.OpenSlide(wsi_path)
    features = []
    batch = []
    for (x, y) in coords:
        tile = slide.read_region((int(x), int(y)), 0, (tile_size, tile_size)).convert('RGB')
        tile_tensor = transform(tile)
        batch.append(tile_tensor)
        if len(batch) == batch_size:
            with torch.no_grad():
                out = model(torch.stack(batch).to(device))
            features.append(out.cpu())
            batch = []
    if batch:
        with torch.no_grad():
            out = model(torch.stack(batch).to(device))
        features.append(out.cpu())
    slide.close()
    if features:
        feats = torch.cat(features, dim=0)
        slide_feat = feats.mean(dim=0)
        return slide_feat.numpy(), feats.numpy()
    return None, None


def process_all_slides(max_slides=None, batch_size=16, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = locations.get_conch_model_path()
    model, transform = create_model_from_pretrained(model_path)
    model.to(device)
    model.eval()

    dataset_dir = locations.get_dataset_dir()
    seg_dir = locations.get_segmentation_output_dir()
    feat_dir = locations.get_features_output_dir()
    os.makedirs(feat_dir, exist_ok=True)

    wsi_files = sorted([f for f in os.listdir(dataset_dir) if f.endswith('.tif')])
    if max_slides is not None:
        wsi_files = wsi_files[:max_slides]

    for fname in wsi_files:
        slide_name = os.path.splitext(fname)[0]
        slide_dir = os.path.join(seg_dir, slide_name)
        coords, tile_size = load_tile_coordinates(slide_dir, slide_name)
        wsi_path = os.path.join(dataset_dir, fname)
        print(f"Processing {fname} with {len(coords)} tiles...")
        slide_feat, tile_feats = extract_features_for_slide(model, transform, wsi_path, coords, tile_size, batch_size, device)
        if slide_feat is None:
            print(f"No features extracted for {fname}")
            continue
        out_path = os.path.join(feat_dir, f"{slide_name}_conch15.npz")
        np.savez_compressed(out_path, slide_feature=slide_feat, tile_features=tile_feats)
        print(f"Saved features to {out_path}")


if __name__ == '__main__':
    process_all_slides()
