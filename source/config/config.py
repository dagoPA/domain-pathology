"""
Centralized configuration file for the pathology scripts.
"""

# -----------------------------------------------------------------------------
# Segmentation Parameters (for segmentation.py)
# -----------------------------------------------------------------------------
SEG_MEDIAN_BLUR_SIZE = 7
SEG_CLOSE_KERNEL_SIZE = 7
SEG_MIN_CONTOUR_AREA = 1000

# -----------------------------------------------------------------------------
# Tiling Parameters (for tiling.py)
# -----------------------------------------------------------------------------
TILE_PATCH_SIZE = 448
TILE_MAGNIFICATION = 20
TILE_OVERLAP = 0
TILE_TISSUE_THRESHOLD = 0.5

# -----------------------------------------------------------------------------
# Feature Extraction Parameters (for feature_extraction.py)
# -----------------------------------------------------------------------------
# The CONCH v1.5 model requires a specific input patch size.
# This should NOT be changed unless you use a different model.
FEAT_PATCH_SIZE = 448

# Batch size for processing tiles. Adjust based on your GPU memory.
FEAT_BATCH_SIZE = 64

# Hugging Face path for the pretrained CONCH v1.5 model.
FEAT_MODEL_CHECKPOINT = "hf_hub:MahmoodLab/conchv1_5"