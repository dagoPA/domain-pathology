"""
This module implements WSI (Whole Slide Image) tissue segmentation
based on an adaptation of the CLAM algorithm, optimized to work
exclusively with the lowest-resolution image (thumbnail).

The process follows an efficient strategy:
1. The lowest-resolution version of the image (the highest-level thumbnail)
   is read from the WSI to ensure maximum speed.
2. Segmentation is performed on this small image using Otsu's method for
   automatic thresholding, making it more robust.
3. The resulting artifacts, such as the low-resolution mask,
   a visualization, and the parameters used, are saved.
"""

import os
import cv2
import numpy as np
import openslide
from matplotlib import pyplot as plt
import glob
import json

# The 'locations' module is expected to be provided by the project's
# configuration to manage input and output paths.
from source.config import locations


# --- Default Parameters for CLAM-like Segmentation ---
# saturation_threshold is removed as we now use Otsu's automatic thresholding.
DEFAULT_PARAMS = {
    "median_blur_size": 7,     # Kernel size for the median blur filter.
    "close_kernel_size": 7,    # Kernel size for the morphological closing operation.
    "min_contour_area": 1000,  # Minimum area for a contour to be considered tissue. Reduced to be less strict.
}

def create_tissue_mask(wsi_path, params=DEFAULT_PARAMS):
    """
    Generates a binary tissue mask from a WSI's thumbnail.

    The process is performed on the lowest-resolution image for efficiency and uses
    Otsu's method for robust, automatic thresholding.

    Args:
        wsi_path (str): Path to the WSI file.
        params (dict): Dictionary with segmentation parameters.

    Returns:
        - np.ndarray: The low-resolution binary tissue mask.
        - np.ndarray: The low-resolution RGB thumbnail image.
        - int: The WSI level used for processing.
    """
    try:
        wsi = openslide.OpenSlide(wsi_path)

        # --- Step 1: Work with the lowest-resolution image ---
        # For maximum efficiency, we select the highest pyramid level
        # (wsi.level_count - 1), which corresponds to the lowest-resolution image.
        level = wsi.level_count - 1
        print(f"Using the lowest available resolution level: {level} for efficient segmentation.")

        # Read the image from the selected level.
        thumb = wsi.read_region((0, 0), level, wsi.level_dimensions[level])
        thumb_rgb = np.array(thumb.convert('RGB'))
        wsi.close()

        # --- Step 2: Segmentation on the low-resolution image ---
        # Convert to HSV color space to use the saturation channel,
        # which is very effective for distinguishing tissue from background.
        hsv_image = cv2.cvtColor(thumb_rgb, cv2.COLOR_RGB2HSV)
        saturation_channel = hsv_image[:, :, 1]

        # Apply a median blur to smooth the image and remove noise.
        blur_size = params.get("median_blur_size", DEFAULT_PARAMS["median_blur_size"])
        blurred_saturation = cv2.medianBlur(saturation_channel, blur_size)

        # --- MODIFICATION: Use Otsu's Binarization ---
        # Instead of a fixed threshold, Otsu's method automatically finds the
        # optimal threshold value from the image histogram. This is much more
        # robust for images with different staining or brightness.
        _, binary_mask = cv2.threshold(
            blurred_saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Use a morphological closing operation to fill small holes in the tissue.
        close_kernel_size = params.get("close_kernel_size", DEFAULT_PARAMS["close_kernel_size"])
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size))
        closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

        # Filter contours by area to remove small artifacts and noise.
        min_area = params.get("min_contour_area", DEFAULT_PARAMS["min_contour_area"])
        contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        low_res_mask = np.zeros_like(closed_mask)
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        cv2.drawContours(low_res_mask, filtered_contours, -1, (255), -1)

        return low_res_mask, thumb_rgb, level

    except openslide.OpenSlideError as e:
        print(f"Error opening WSI file {wsi_path}: {e}")
        return None, None, None
    except Exception as e:
        print(f"An unexpected error occurred while processing {wsi_path}: {e}")
        return None, None, None

def visualize_segmentation(wsi_path, low_res_mask, thumb_rgb, level, output_path):
    """
    Visualizes and saves the low-resolution segmentation result using Matplotlib.
    """
    masked_thumb = thumb_rgb.copy()
    masked_thumb[low_res_mask == 0] = 0  # Black background

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    wsi_filename = os.path.basename(wsi_path)
    fig.suptitle(f"Tissue Segmentation for: {wsi_filename}\n(Processed at level {level})", fontsize=16)

    axes[0].imshow(thumb_rgb)
    axes[0].set_title("Original Thumbnail")
    axes[0].axis('off')

    axes[1].imshow(low_res_mask, cmap='gray')
    axes[1].set_title("Segmentation Mask")
    axes[1].axis('off')

    axes[2].imshow(masked_thumb)
    axes[2].set_title("Tissue on Black Background")
    axes[2].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Visualization saved to: {output_path}")

def process_all_slides(params=DEFAULT_PARAMS, max_slides=None):
    """
    Processes all WSI .tif files and saves the segmentation artifacts
    based on the thumbnail.

    Args:
        params (dict): Parameters for the segmentation algorithm.
        max_slides (int, optional): Maximum number of slides to process. Defaults to None.
    """
    wsi_dir = locations.get_dataset_dir()
    output_base_dir = locations.get_segmentation_output_dir()
    os.makedirs(output_base_dir, exist_ok=True)

    wsi_files = sorted(glob.glob(os.path.join(wsi_dir, "*.tif")))
    if not wsi_files:
        print(f"No .tif files found in the directory: {wsi_dir}")
        return

    if max_slides is not None:
        wsi_files = wsi_files[:max_slides]
        print(f"Processing a subset of {len(wsi_files)} slides.")

    for wsi_path in wsi_files:
        print(f"\n--- Processing: {os.path.basename(wsi_path)} ---")

        low_res_mask, thumb_rgb, level = create_tissue_mask(wsi_path, params)

        if low_res_mask is not None:
            slide_name = os.path.splitext(os.path.basename(wsi_path))[0]
            slide_output_dir = os.path.join(output_base_dir, slide_name)
            os.makedirs(slide_output_dir, exist_ok=True)

            # --- Define output paths ---
            vis_path = os.path.join(slide_output_dir, f"{slide_name}_visualization.png")
            low_mask_path = os.path.join(slide_output_dir, f"{slide_name}_low_res_mask.png")
            params_path = os.path.join(slide_output_dir, f"{slide_name}_params.json")

            # --- Save artifacts ---
            visualize_segmentation(wsi_path, low_res_mask, thumb_rgb, level, vis_path)

            # Save the low-resolution mask as a PNG for quick visual inspection.
            cv2.imwrite(low_mask_path, low_res_mask)
            print(f"Low-resolution mask saved to: {low_mask_path}")

            # Save the parameters used for the segmentation, including the level.
            params_to_save = params.copy()
            params_to_save['level_used'] = level
            with open(params_path, 'w') as f:
                json.dump(params_to_save, f, indent=4)
            print(f"Parameters saved to: {params_path}")
        else:
            print(f"Skipping artifact generation for {os.path.basename(wsi_path)} due to a processing error.")

    print("\n--- All slides have been processed. ---")

if __name__ == '__main__':
    print("--- Running segmentation script in standalone mode ---")
    # To run, ensure your project's 'locations' configuration is set up correctly.
    # The script will process the first 2 slides by default.
    # Change max_slides to None to process all of them.
    process_all_slides(max_slides=2)
    print("\n--- Standalone script execution finished. ---")
