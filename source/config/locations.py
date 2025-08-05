"""
Configuration module for domain-pathology project.
Contains global variables and paths used throughout the project.
"""

import os

# --- Primary Paths (Configured for a specific server environment) ---
PRIMARY_BASE_DIR = "/home/dagopa/projects/domain-pathology"
PRIMARY_DATASET_DIR = "/home/dagopa/data/CAMELYON17/WSI"
CONCH_MODEL_PATH = "/autofs/space/crater_001/tools/wsi_encoders/conch_v15.bin"

# --- Project-specific Subdirectories ---
OUTPUTS_DIR = os.path.join(PRIMARY_BASE_DIR, "outputs")
VISUALIZATIONS_OUTPUT_DIR = os.path.join(OUTPUTS_DIR, "visualizations")
PREPROCESSING_OUTPUT_DIR = os.path.join(OUTPUTS_DIR, "preprocessing")
TILING_OUTPUT_DIR = os.path.join(PREPROCESSING_OUTPUT_DIR, "tiles")
SEGMENTATION_OUTPUT_DIR = os.path.join(PREPROCESSING_OUTPUT_DIR, "segmentations")
FEATURES_OUTPUT_DIR = os.path.join(OUTPUTS_DIR, "features")
CONCH_FEATURES_OUTPUT_DIR = os.path.join(FEATURES_OUTPUT_DIR, "conchv_15")
TITAN_FEATURES_OUTPUT_DIR = os.path.join(FEATURES_OUTPUT_DIR, "titan")

CACHE_DIR = os.path.join(PRIMARY_BASE_DIR, "cache")
LABELS_DIR = os.path.join(PRIMARY_BASE_DIR, "labels")


def _create_dir_with_fallback(primary_path, fallback_subdir_name):
    """
    Attempts to create and return the primary_path.
    If it fails (e.g., due to permissions), it creates and returns a local fallback directory.
    """
    try:
        os.makedirs(primary_path, exist_ok=True)
        return primary_path
    except (OSError, PermissionError) as e:
        print(f"Warning: Could not create configured directory '{primary_path}': {e}")
        # To handle Windows paths correctly in fallback
        fallback_path_parts = fallback_subdir_name.split('/')
        fallback_dir = os.path.join(os.getcwd(), *fallback_path_parts)
        print(f"Using fallback directory: {fallback_dir}")
        os.makedirs(fallback_dir, exist_ok=True)
        return fallback_dir


def get_visualizations_output_dir():
    """Get the configured output directory for visualizations."""
    return _create_dir_with_fallback(VISUALIZATIONS_OUTPUT_DIR, 'outputs/visualizations')

def get_tiling_output_dir():
    """Get the configured output directory for tiling results."""
    return _create_dir_with_fallback(TILING_OUTPUT_DIR, 'outputs/preprocessing/tiles')

def get_segmentation_output_dir():
    """Get the configured output directory for segmentation results."""
    return _create_dir_with_fallback(SEGMENTATION_OUTPUT_DIR, 'outputs/preprocessing/segmentations')

def get_conch_features_output_dir():
    """Get the configured features output directory for CONCH patch features."""
    return _create_dir_with_fallback(CONCH_FEATURES_OUTPUT_DIR, 'outputs/features/conchv_15')

def get_titan_features_output_dir():
    """Get the configured features output directory for TITAN slide features."""
    return _create_dir_with_fallback(TITAN_FEATURES_OUTPUT_DIR, 'outputs/features/titan')

def get_cache_dir():
    """Get the configured cache directory."""
    return _create_dir_with_fallback(CACHE_DIR, 'cache')

def get_dataset_dir():
    """Get the configured dataset directory."""
    return _create_dir_with_fallback(PRIMARY_DATASET_DIR, 'datasets/CAMELYON17/WSI')

def get_labels_dir():
    """Get the configured labels directory."""
    return _create_dir_with_fallback(LABELS_DIR, 'labels')

def get_labels_csv_path():
    """Get the path to the CAMELYON17 labels CSV file."""
    labels_dir = get_labels_dir()
    return os.path.join(labels_dir, 'camelyon17-labels.csv')

def get_conch_model_path():
    """Get the path to the CONCH model file."""
    global CONCH_MODEL_PATH
    return CONCH_MODEL_PATH

def set_conch_model_path(new_path):
    """Set a new path for the CONCH model file."""
    global CONCH_MODEL_PATH
    CONCH_MODEL_PATH = new_path
    print(f"CONCH model path updated to: {CONCH_MODEL_PATH}")
