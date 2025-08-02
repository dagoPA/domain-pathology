import h5py
import numpy as np

# Abrir features_conch.h5
with h5py.File("/home/dagopa/projects/domain-pathology/outputs/segmentation/patient_000_node_3/features_conch.h5", 'r') as f:
    features = f['features'][:]
    coords = f['coords'][:]
    print(f"features_conch.h5 - features shape: {features.shape}")
    print(f"features_conch.h5 - coords shape: {coords.shape}")

# Abrir feature_slide_titan.npy
data = np.load("/home/dagopa/projects/domain-pathology/outputs/segmentation/patient_000_node_3/feature_slide_titan.npy")
print(f"feature_slide_titan.npy shape: {data.shape}")

print('completed')