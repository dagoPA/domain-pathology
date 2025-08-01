"""
Centralized configuration file for the pathology scripts.
"""

# -----------------------------------------------------------------------------
# Segmentation Parameters (for segmentation.py)
# -----------------------------------------------------------------------------
# Kernel size for the median blur filter.
SEG_MEDIAN_BLUR_SIZE = 7

# Kernel size for the morphological closing operation.
SEG_CLOSE_KERNEL_SIZE = 7

# Minimum area in pixels (at the thumbnail level) for a contour to be
# considered valid tissue.
SEG_MIN_CONTOUR_AREA = 1000


# -----------------------------------------------------------------------------
# Tiling Parameters (for tiling.py)
# -----------------------------------------------------------------------------
# Final size of each patch in pixels (e.g., 256x256).
TILE_PATCH_SIZE = 256

# Target magnification at which patches are extracted (e.g., 20x).
TILE_MAGNIFICATION = 20

# Pixel overlap between adjacent patches.
TILE_OVERLAP = 0

# Minimum proportion of tissue that a patch must contain to be considered
# valid (e.g., 0.5 = 50% of the patch area).
TILE_TISSUE_THRESHOLD = 0.5