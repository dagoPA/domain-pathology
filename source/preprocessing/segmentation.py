"""
This module implements WSI tissue segmentation based on the CLAM algorithm.
It processes slides, generates tissue masks, and extracts tile coordinates efficiently.
The tile coordinate generation is optimized to only search within tissue regions.
The full-scale mask is generated in memory but is not saved to disk.
"""

import os
import cv2
import numpy as np
import openslide
from matplotlib import pyplot as plt
import glob
import json

from source.config import locations


# --- Default Parameters for CLAM Segmentation & Tiling ---
DEFAULT_PARAMS = {
    # Segmentation parameters
    "level": 6,
    "saturation_threshold": 20,
    "median_blur_size": 7,
    "close_kernel_size": 7,
    "min_contour_area": 5000,

    # Tiling parameters
    "tile_size": 256,
    "tile_min_tissue_fraction": 0.5
}

def create_tissue_mask(wsi_path, params=DEFAULT_PARAMS):
    """
    Generates a binary tissue mask from a WSI using a CLAM-like algorithm.

    Args:
        wsi_path (str): Path to the WSI file.
        params (dict): Dictionary of segmentation parameters.

    Returns:
        - np.ndarray: The full-scale binary tissue mask (at level 0).
        - np.ndarray: The low-resolution binary tissue mask for visualization.
        - np.ndarray: The downsampled RGB thumbnail image.
        - int: The WSI level used for processing.
    """
    try:
        wsi = openslide.OpenSlide(wsi_path)

        level = params.get("level", DEFAULT_PARAMS["level"])
        if level >= wsi.level_count:
            level = wsi.level_count - 1
            print(f"Warning: Level {params['level']} not available. Using level {level}.")

        thumb = wsi.read_region((0, 0), level, wsi.level_dimensions[level])
        thumb_rgb = np.array(thumb.convert('RGB'))

        hsv_image = cv2.cvtColor(thumb_rgb, cv2.COLOR_RGB2HSV)
        saturation_channel = hsv_image[:, :, 1]

        blur_size = params.get("median_blur_size", DEFAULT_PARAMS["median_blur_size"])
        blurred_saturation = cv2.medianBlur(saturation_channel, blur_size)
        sat_thresh = params.get("saturation_threshold", DEFAULT_PARAMS["saturation_threshold"])
        _, binary_mask = cv2.threshold(
            blurred_saturation, sat_thresh, 255, cv2.THRESH_BINARY
        )

        close_kernel_size = params.get("close_kernel_size", DEFAULT_PARAMS["close_kernel_size"])
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size))
        closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = params.get("min_contour_area", DEFAULT_PARAMS["min_contour_area"])
        low_res_mask = np.zeros_like(closed_mask)
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        cv2.drawContours(low_res_mask, filtered_contours, -1, (255), -1)

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


def get_tile_coordinates_optimized(full_mask, low_res_mask, downsample_factor, params=DEFAULT_PARAMS):
    """
    Generates valid tile coordinates from a full-scale tissue mask by searching
    only within the bounding boxes of tissue regions found at low resolution.

    Args:
        full_mask (np.ndarray): The full-scale binary tissue mask.
        low_res_mask (np.ndarray): The low-resolution binary tissue mask.
        downsample_factor (float): The downsampling factor from level 0 to the low-res level.
        params (dict): Dictionary containing tiling parameters.

    Returns:
        list: A list of [x, y] coordinates for the top-left corner of each valid tile.
    """
    tile_size = params.get("tile_size", DEFAULT_PARAMS["tile_size"])
    min_frac = params.get("tile_min_tissue_fraction", DEFAULT_PARAMS["tile_min_tissue_fraction"])
    min_tissue_pixels = tile_size * tile_size * min_frac

    coordinates_set = set()

    # Find contours on the low-resolution mask to identify tissue regions
    contours, _ = cv2.findContours(low_res_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Get the bounding box of the contour at low resolution
        x_low, y_low, w_low, h_low = cv2.boundingRect(contour)

        # Scale the bounding box to full resolution (level 0)
        x_start = int(x_low * downsample_factor)
        y_start = int(y_low * downsample_factor)
        x_end = int((x_low + w_low) * downsample_factor)
        y_end = int((y_low + h_low) * downsample_factor)

        # Align the start coordinates to the tile grid
        x_start_grid = x_start - (x_start % tile_size)
        y_start_grid = y_start - (y_start % tile_size)

        # Iterate only within the scaled bounding box
        for y in range(y_start_grid, y_end, tile_size):
            for x in range(x_start_grid, x_end, tile_size):
                tile_region = full_mask[y:y + tile_size, x:x + tile_size]
                if tile_region.shape[0] != tile_size or tile_region.shape[1] != tile_size:
                    continue

                if cv2.countNonZero(tile_region) >= min_tissue_pixels:
                    coordinates_set.add((x, y))

    return sorted(list(coordinates_set))


def get_tile_coordinates_integral(full_mask, low_res_mask, downsample_factor, params=DEFAULT_PARAMS):
    """Faster tile coordinate generation using integral images.

    The search is restricted to the tissue bounding boxes obtained at low
    resolution and uses an integral image to quickly compute the tissue
    fraction for each tile.

    Args:
        full_mask (np.ndarray): Full resolution binary mask (0/255).
        low_res_mask (np.ndarray): Low resolution mask used to locate tissue.
        downsample_factor (float): Ratio between full and low resolution levels.
        params (dict): Segmentation and tiling parameters.

    Returns:
        list: Sorted list of valid tile ``(x, y)`` coordinates at level 0.
    """
    tile_size = params.get("tile_size", DEFAULT_PARAMS["tile_size"])
    min_frac = params.get("tile_min_tissue_fraction", DEFAULT_PARAMS["tile_min_tissue_fraction"])
    min_tissue_pixels = tile_size * tile_size * min_frac * 255

    # Integral image for O(1) region sums. Convert mask to 0/1 then compute integral
    integral = cv2.integral((full_mask > 0).astype(np.uint8), sdepth=cv2.CV_32S)

    coordinates = []

    contours, _ = cv2.findContours(low_res_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x_low, y_low, w_low, h_low = cv2.boundingRect(contour)

        x_start = int(x_low * downsample_factor)
        y_start = int(y_low * downsample_factor)
        x_end = int((x_low + w_low) * downsample_factor)
        y_end = int((y_low + h_low) * downsample_factor)

        x_start_grid = x_start - (x_start % tile_size)
        y_start_grid = y_start - (y_start % tile_size)

        for y in range(y_start_grid, y_end - tile_size + 1, tile_size):
            y2 = y + tile_size
            for x in range(x_start_grid, x_end - tile_size + 1, tile_size):
                x2 = x + tile_size
                region_sum = (integral[y2, x2] - integral[y, x2] - integral[y2, x] + integral[y, x])
                if region_sum >= min_tissue_pixels:
                    coordinates.append((x, y))

    return coordinates


def visualize_segmentation(wsi_path, mask, thumb_rgb, level, output_path):
    """
    Visualizes and saves the segmentation result.
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
    Processes all WSI TIF files, saves artifacts, and extracts tile coordinates.
    """
    wsi_dir = locations.get_dataset_dir()
    output_base_dir = locations.get_segmentation_output_dir()
    os.makedirs(output_base_dir, exist_ok=True)

    wsi_files = sorted(glob.glob(os.path.join(wsi_dir, "*.tif")))
    if not wsi_files:
        print(f"No .tif files found in the dataset directory: {wsi_dir}")
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
            mask_path = os.path.join(slide_output_dir, f"{slide_name}_mask.png")
            coords_path = os.path.join(slide_output_dir, f"{slide_name}_tile_coords.json")
            params_path = os.path.join(slide_output_dir, f"{slide_name}_params.json")

            # --- Save artifacts ---
            visualize_segmentation(wsi_path, low_res_mask, thumb_rgb, level, vis_path)
            cv2.imwrite(mask_path, low_res_mask)
            print(f"Saved low-resolution mask to: {mask_path}")

            # Get downsample factor for coordinate mapping
            try:
                with openslide.OpenSlide(wsi_path) as slide:
                    downsample_factor = slide.level_downsamples[level]
            except openslide.OpenSlideError as e:
                print(f"Could not reopen slide to get downsample factor: {e}")
                continue

            # Get and save tile coordinates using the integral image implementation
            print("Extracting tile coordinates (fast)...")
            tile_coords = get_tile_coordinates_integral(full_mask, low_res_mask, downsample_factor, params)

            coords_data = {
                "tile_size": params.get("tile_size", DEFAULT_PARAMS["tile_size"]),
                "tile_level": 0,
                "num_tiles": len(tile_coords),
                "coordinates": tile_coords
            }
            with open(coords_path, 'w') as f:
                json.dump(coords_data, f, indent=4)
            print(f"Saved {len(tile_coords)} tile coordinates to: {coords_path}")

            # Save coordinates also as compressed NumPy for faster loading
            npz_path = os.path.join(slide_output_dir, f"{slide_name}_tile_coords.npz")
            np.savez_compressed(npz_path,
                                coordinates=np.array(tile_coords, dtype=np.int32),
                                tile_size=params.get("tile_size", DEFAULT_PARAMS["tile_size"]),
                                tile_level=0)
            print(f"Saved coordinates in compressed format to: {npz_path}")

            params_to_save = params.copy()
            params_to_save['level_used'] = level
            with open(params_path, 'w') as f:
                json.dump(params_to_save, f, indent=4)
            print(f"Saved parameters to: {params_path}")
        else:
            print(f"Skipping artifact generation for {os.path.basename(wsi_path)} due to a processing error.")

    print("\n--- All slides processed successfully. ---")

if __name__ == '__main__':
    print("--- Running segmentation script in standalone mode ---")
    process_all_slides(max_slides=2)

    print("\n--- Standalone script execution finished. ---")
