"""
Simple configuration file for domain-pathology project.
"""

# Simple configuration - keeping it minimal as requested

# Feature extraction configuration
PATCH_SIZE = 256  # Configurable patch size for feature extraction

# Tissue segmentation configuration - default parameters
USE_TISSUE_SEGMENTATION = True
SEG_LEVEL = 9
STHRESH = 20
STHRESH_UP = 255
MTHRESH = 7
CLOSE = 0
USE_OTSU = False
FILTER_PARAMS = {'a_t': 100, 'a_h': 16, 'max_n_holes': 8}
REF_PATCH_SIZE = 512
EXCLUDE_IDS = []
