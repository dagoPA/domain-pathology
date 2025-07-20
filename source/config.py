"""
Configuration module for domain-pathology project.
Contains global variables and paths used throughout the project.
"""

import os

# Base results directory
RESULTS_DIR = "/autofs/space/crater_001/"

# Project-specific paths
PROJECT_BASE_DIR = os.path.join(RESULTS_DIR, "projects/micropath/domain-pathology")
FIGURES_DIR = os.path.join(PROJECT_BASE_DIR, "figures")
CACHE_DIR = os.path.join(PROJECT_BASE_DIR, "cache")
DATASET_DIR = os.path.join(PROJECT_BASE_DIR, "dataset")

# Ensure directories exist when imported
def ensure_directories():
    """Create necessary directories if they don't exist."""
    try:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        os.makedirs(CACHE_DIR, exist_ok=True)
        os.makedirs(DATASET_DIR, exist_ok=True)
        return FIGURES_DIR
    except (OSError, PermissionError) as e:
        # Fallback to local outputs directory if configured path is not accessible
        print(f"Warning: Could not create configured directory {FIGURES_DIR}: {e}")
        fallback_dir = os.path.join(os.getcwd(), 'outputs')
        fallback_cache_dir = os.path.join(os.getcwd(), 'cache')
        fallback_dataset_dir = os.path.join(os.getcwd(), 'dataset')
        print(f"Using fallback directory: {fallback_dir}")
        print(f"Using fallback cache directory: {fallback_cache_dir}")
        print(f"Using fallback dataset directory: {fallback_dataset_dir}")
        os.makedirs(fallback_dir, exist_ok=True)
        os.makedirs(fallback_cache_dir, exist_ok=True)
        os.makedirs(fallback_dataset_dir, exist_ok=True)
        return fallback_dir

# For backward compatibility, also provide the output directory
def get_output_dir():
    """Get the configured output directory for analysis results."""
    return ensure_directories()

def get_cache_dir():
    """Get the configured cache directory."""
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        return CACHE_DIR
    except (OSError, PermissionError) as e:
        # Fallback to local cache directory if configured path is not accessible
        print(f"Warning: Could not create configured cache directory {CACHE_DIR}: {e}")
        fallback_cache_dir = os.path.join(os.getcwd(), 'cache')
        print(f"Using fallback cache directory: {fallback_cache_dir}")
        os.makedirs(fallback_cache_dir, exist_ok=True)
        return fallback_cache_dir

def get_dataset_dir():
    """Get the configured dataset directory."""
    try:
        os.makedirs(DATASET_DIR, exist_ok=True)
        return DATASET_DIR
    except (OSError, PermissionError) as e:
        # Fallback to local dataset directory if configured path is not accessible
        print(f"Warning: Could not create configured dataset directory {DATASET_DIR}: {e}")
        fallback_dataset_dir = os.path.join(os.getcwd(), 'dataset')
        print(f"Using fallback dataset directory: {fallback_dataset_dir}")
        os.makedirs(fallback_dataset_dir, exist_ok=True)
        return fallback_dataset_dir
