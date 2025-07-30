"""
This module implements WSI tissue segmentation based on the CLAM algorithm.
It processes slides from the configured dataset directory, generates tissue masks,
saves full-scale masks, and extracts coordinates for valid tissue tiles.
"""

import os
import cv2
import numpy as np
import openslide
from matplotlib import pyplot as plt
import glob
import json

# Import project-specific configuration to get data paths
# Assuming 'source.config' is a valid module in your project structure.
# If not, you might need to adjust the import or paths.
try:
    from source.config import locations
except ImportError:
    # Provide a fallback for standalone execution if config is not found
    class MockLocations:
        def get_dataset_dir(self):
            return "path/to/your/slides"
        def get_segmentation_output_dir(self):
            return "output/segmentation"
    locations = MockLocations()
    print("Warning: 'source.config.locations' not found. Using mock paths.")


# --- Default Parameters for CLAM Segmentation & Tiling ---
# These can be tuned for different datasets.
DEFAULT_PARAMS = {
    # Segmentation parameters
    "level": 6,                 # WSI level to use for segmentation (lower resolution).
    "saturation_threshold": 20, # Threshold for the saturation channel in HSV.
    "median_blur_size": 7,      # Kernel size for median blurring to remove noise.
    "close_kernel_size": 7,     # Kernel size for morphological closing to fill holes.
    "min_contour_area": 5000,   # Minimum pixel area for a contour to be considered tissue.

    # Tiling parameters
    "tile_size": 256,               # Size of the tiles to extract (at level 0).
    "tile_min_tissue_fraction": 0.5 # Minimum fraction of tissue required for a tile to be valid.
}

def create_tissue_mask(wsi_path, params=DEFAULT_PARAMS):
    """
    Generates a binary tissue mask from a WSI using a CLAM-like algorithm.

    This function creates both a low-resolution mask for visualization and a
    full-resolution mask (level 0) for tiling.

    Args:
        wsi_path (str): Path to the WSI file.
        params (dict): Dictionary of segmentation parameters.

    Returns:
        - np.ndarray: The full-scale binary tissue mask (at level 0).
        - np.ndarray: The low-resolution binary tissue mask used for visualization.
        - np.ndarray: The downsampled RGB thumbnail image used for segmentation.
        - int: The WSI level used for processing the thumbnail.
    """
    try:
        wsi = openslide.OpenSlide(wsi_path)

        # Use a low-resolution level for speed, as defined in params
        level = params.get("level", DEFAULT_PARAMS["level"])
        if level >= wsi.level_count:
            level = wsi.level_count - 1
            print(f"Warning: Level {params['level']} not available. Using level {level}.")

        # 1. Read WSI at a downsampled resolution (thumbnail)
        thumb = wsi.read_region((0, 0), level, wsi.level_dimensions[level])
        thumb_rgb = np.array(thumb.convert('RGB'))

        # 2. Convert from RGB to HSV color space
        hsv_image = cv2.cvtColor(thumb_rgb, cv2.COLOR_RGB2HSV)
        saturation_channel = hsv_image[:, :, 1]

        # 3. Median blurring to smooth edges
        blur_size = params.get("median_blur_size", DEFAULT_PARAMS["median_blur_size"])
        blurred_saturation = cv2.medianBlur(saturation_channel, blur_size)

        # 4. Compute binary mask via thresholding the saturation channel
        sat_thresh = params.get("saturation_threshold", DEFAULT_PARAMS["saturation_threshold"])
        _, binary_mask = cv2.threshold(
            blurred_saturation, sat_thresh, 255, cv2.THRESH_BINARY
        )

        # 5. Morphological closing to fill small gaps and holes
        close_kernel_size = params.get("close_kernel_size", DEFAULT_PARAMS["close_kernel_size"])
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size))
        closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

        # 6. Filter contours based on area to remove small artifacts
        contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = params.get("min_contour_area", DEFAULT_PARAMS["min_contour_area"])
        low_res_mask = np.zeros_like(closed_mask)
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        cv2.drawContours(low_res_mask, filtered_contours, -1, (255), -1)

        # 7. Resize the low-resolution mask to full scale (level 0)
        full_dims = wsi.level_dimensions[0]
        full_mask = cv2.resize(low_res_mask, (full_dims[0], full_dims[1]), interpolation=cv2.INTER_NEAREST)

        wsi.close()
        return full_mask, low_res_mask, thumb_rgb, level

    except openslide.OpenSlideError as e:
        print(f"Error opening WSI file {wsi_path}: {e}")
        return None, None, None, None
    except Exception as e:
        print(f"An unexpected error occurred while processing {wsi_path}: {e}")
        return None, None, None, None


def get_tile_coordinates(mask, params=DEFAULT_PARAMS):
    """
    Generates valid tile coordinates from a full-scale tissue mask.

    Args:
        mask (np.ndarray): The full-scale binary tissue mask (values 0 or 255).
        params (dict): Dictionary containing tiling parameters like 'tile_size'
                       and 'tile_min_tissue_fraction'.

    Returns:
        list: A list of [x, y] coordinates for the top-left corner of each valid tile.
    """
    tile_size = params.get("tile_size", DEFAULT_PARAMS["tile_size"])
    min_frac = params.get("tile_min_tissue_fraction", DEFAULT_PARAMS["tile_min_tissue_fraction"])

    coordinates = []
    mask_h, mask_w = mask.shape
    min_tissue_pixels = (tile_size * tile_size * min_frac)

    for y in range(0, mask_h, tile_size):
        for x in range(0, mask_w, tile_size):
            # Ensure the tile is fully within the mask bounds
            if x + tile_size > mask_w or y + tile_size > mask_h:
                continue

            tile_region = mask[y:y+tile_size, x:x+tile_size]

            # Count non-zero (tissue) pixels. cv2.countNonZero is fast.
            tissue_pixels = cv2.countNonZero(tile_region)

            if tissue_pixels >= min_tissue_pixels:
                coordinates.append([x, y])

    return coordinates


def visualize_segmentation(wsi_path, mask, thumb_rgb, level, output_path):
    """
    Visualizes and saves the segmentation result, showing the tissue on a black background.

    Args:
        wsi_path (str): Path to the original WSI file.
        mask (np.ndarray): The low-resolution binary tissue mask.
        thumb_rgb (np.ndarray): The downsampled RGB thumbnail.
        level (int): The WSI level used for processing.
        output_path (str): Path to save the visualization image.
    """
    masked_thumb = thumb_rgb.copy()
    masked_thumb[mask == 0] = 0

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    wsi_filename = os.path.basename(wsi_path)
    fig.suptitle(f"Tissue Segmentation for: {wsi_filename}\n(Processed at level {level})", fontsize=16)

    axes[0].imshow(thumb_rgb)
    axes[0].set_title("Original Thumbnail")
    axes[0].axis('off')

    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title("Segmentation Mask")
    axes[1].axis('off')

    axes[2].imshow(masked_thumb)
    axes[2].set_title("Tissue on Black Background")
    axes[2].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Saved visualization to: {output_path}")


def process_all_slides(params=DEFAULT_PARAMS, max_slides=None):
    """
    Processes all WSI TIF files, saves masks, visualizations, and tile coordinates.

    Args:
        params (dict): Parameters for the segmentation and tiling algorithms.
        max_slides (int, optional): Maximum number of slides to process. Defaults to None (all slides).
    """
    wsi_dir = locations.get_dataset_dir()
    output_base_dir = locations.get_segmentation_output_dir()
    os.makedirs(output_base_dir, exist_ok=True)

    print(f"Searching for WSI files in: {wsi_dir}")
    wsi_files = sorted(glob.glob(os.path.join(wsi_dir, "*.tif")))

    if not wsi_files:
        print("No .tif files found in the dataset directory.")
        return

    if max_slides is not None:
        wsi_files = wsi_files[:max_slides]
        print(f"Processing a subset of {len(wsi_files)} slides.")

    for wsi_path in wsi_files:
        print(f"\n--- Processing: {os.path.basename(wsi_path)} ---")

        full_mask, low_res_mask, thumb_rgb, level = create_tissue_mask(wsi_path, params)

        if full_mask is not None:
            slide_name = os.path.splitext(os.path.basename(wsi_path))[0]
            slide_output_dir = os.path.join(output_base_dir, slide_name)
            os.makedirs(slide_output_dir, exist_ok=True)

            # --- Define output paths ---
            vis_path = os.path.join(slide_output_dir, f"{slide_name}_segmentation.png")
            low_res_mask_path = os.path.join(slide_output_dir, f"{slide_name}_mask.png")
            full_mask_path = os.path.join(slide_output_dir, f"{slide_name}_full_mask.png")
            coords_path = os.path.join(slide_output_dir, f"{slide_name}_tile_coords.json")
            params_path = os.path.join(slide_output_dir, f"{slide_name}_params.json")

            # --- Save artifacts ---
            # 1. Save the visualization plot
            visualize_segmentation(wsi_path, low_res_mask, thumb_rgb, level, vis_path)

            # 2. Save the low-resolution binary mask
            cv2.imwrite(low_res_mask_path, low_res_mask)
            print(f"Saved low-resolution mask to: {low_res_mask_path}")

            # 3. Save the full-scale binary mask
            print("Saving full-scale mask... (This might take a moment)")
            cv2.imwrite(full_mask_path, full_mask)
            print(f"Saved full-scale mask to: {full_mask_path}")

            # 4. Get and save tile coordinates
            print("Extracting tile coordinates...")
            tile_coords = get_tile_coordinates(full_mask, params)
            coords_data = {
                "tile_size": params.get("tile_size", DEFAULT_PARAMS["tile_size"]),
                "tile_level": 0, # Coordinates are for level 0
                "num_tiles": len(tile_coords),
                "coordinates": tile_coords
            }
            with open(coords_path, 'w') as f:
                json.dump(coords_data, f, indent=4)
            print(f"Saved {len(tile_coords)} tile coordinates to: {coords_path}")

            # 5. Save the parameters to a JSON file
            params_to_save = params.copy()
            params_to_save['level_used'] = level
            with open(params_path, 'w') as f:
                json.dump(params_to_save, f, indent=4)
            print(f"Saved parameters to: {params_path}")
        else:
            print(f"Skipping artifact generation for {os.path.basename(wsi_path)} due to a processing error.")

    print("\n--- All slides processed successfully. ---")

# --- Main block for standalone execution and debugging ---
if __name__ == '__main__':
    print("--- Running segmentation script in standalone mode ---")

    # Ensure the output directory exists for the test run
    if isinstance(locations, MockLocations):
        print("Using 'output/segmentation' as the base directory for results.")
        os.makedirs(locations.get_segmentation_output_dir(), exist_ok=True)
        print("Please ensure you have .tif files in 'path/to/your/slides' or update the path in the script.")

    # By default, it runs with the default parameters on a small subset of slides.
    print("\nUsing default parameters for segmentation and tiling.")
    process_all_slides(max_slides=2)

    print("\n--- Standalone script execution finished. ---")
