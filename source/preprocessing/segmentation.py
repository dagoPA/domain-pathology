"""
This module implements WSI tissue segmentation based on the CLAM algorithm.
It processes slides from the configured dataset directory, generates tissue masks,
and saves visualizations and parameters.
"""

import os
import cv2
import numpy as np
import openslide
from matplotlib import pyplot as plt
import glob
import json

# Import project-specific configuration to get data paths
from source.config import locations

# --- Default Parameters for CLAM Segmentation ---
# These can be tuned for different datasets.
DEFAULT_PARAMS = {
    "level": 6,                 # WSI level to use for segmentation (lower resolution).
    "saturation_threshold": 20, # Threshold for the saturation channel in HSV.
    "median_blur_size": 7,      # Kernel size for median blurring to remove noise.
    "close_kernel_size": 7,     # Kernel size for morphological closing to fill holes.
    "min_contour_area": 5000    # Minimum pixel area for a contour to be considered tissue.
}

def create_tissue_mask(wsi_path, params=DEFAULT_PARAMS):
    """
    Generates a binary tissue mask from a WSI using a CLAM-like algorithm.

    Args:
        wsi_path (str): Path to the WSI file.
        params (dict): Dictionary of segmentation parameters.

    Returns:
            - np.ndarray: The final binary tissue mask.
            - np.ndarray: The downsampled RGB thumbnail image used for segmentation.
            - int: The WSI level used for processing.
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
        final_mask = np.zeros_like(closed_mask)

        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

        # Draw the filtered contours to create the final, clean mask
        cv2.drawContours(final_mask, filtered_contours, -1, (255), -1)

        wsi.close()
        return final_mask, thumb_rgb, level

    except openslide.OpenSlideError as e:
        print(f"Error opening WSI file {wsi_path}: {e}")
        return None, None, None
    except Exception as e:
        print(f"An unexpected error occurred while processing {wsi_path}: {e}")
        return None, None, None


def visualize_segmentation(wsi_path, mask, thumb_rgb, level, output_path):
    """
    Visualizes and saves the segmentation result, showing the tissue on a black background.

    Args:
        wsi_path (str): Path to the original WSI file.
        mask (np.ndarray): The binary tissue mask.
        thumb_rgb (np.ndarray): The downsampled RGB thumbnail.
        level (int): The WSI level used for processing.
        output_path (str): Path to save the visualization image.
    """
    # Create a masked version of the thumbnail where the background is black
    masked_thumb = thumb_rgb.copy()
    masked_thumb[mask == 0] = 0

    # Create the plot
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
    plt.show()
    plt.close(fig) # Close the figure to free up memory
    print(f"Saved visualization to: {output_path}")


def process_all_slides(params=DEFAULT_PARAMS, max_slides=None):
    """
    Processes all WSI TIF files in the dataset directory defined in locations.py.

    Args:
        params (dict): Parameters for the segmentation algorithm.
        max_slides (int, optional): Maximum number of slides to process. Defaults to None (all slides).
    """
    wsi_dir = locations.get_dataset_dir()
    # Use the new centralized function for the output directory
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

        mask, thumb_rgb, level = create_tissue_mask(wsi_path, params)

        if mask is not None:
            slide_name = os.path.splitext(os.path.basename(wsi_path))[0]

            # Define output paths for this slide
            slide_output_dir = os.path.join(output_base_dir, slide_name)
            os.makedirs(slide_output_dir, exist_ok=True)

            vis_path = os.path.join(slide_output_dir, f"{slide_name}_segmentation.png")
            mask_path = os.path.join(slide_output_dir, f"{slide_name}_mask.png")
            params_path = os.path.join(slide_output_dir, f"{slide_name}_params.json")

            # Save the visualization plot
            visualize_segmentation(wsi_path, mask, thumb_rgb, level, vis_path)

            # Save the binary mask as an image
            cv2.imwrite(mask_path, mask)
            print(f"Saved mask to: {mask_path}")

            # Save the parameters to a human-readable JSON file
            params_to_save = params.copy()
            params_to_save['level_used'] = level
            with open(params_path, 'w') as f:
                json.dump(params_to_save, f, indent=4)
            print(f"Saved parameters to: {params_path}")
        else:
            print(f"Skipping visualization for {os.path.basename(wsi_path)} due to a processing error.")

    print("\n--- All slides processed successfully. ---")

# --- Main block for standalone execution and debugging ---
if __name__ == '__main__':
    """
    This block allows the script to be run directly from the command line
    for debugging or standalone processing.
    """
    print("--- Running segmentation script in standalone mode ---")

    # To test with custom parameters, you can uncomment and modify this dictionary.
    # For example, to make the segmentation more sensitive:
    # custom_params = {
    #     "level": 6,
    #     "saturation_threshold": 15,  # Lower threshold to capture more tissue
    #     "median_blur_size": 7,
    #     "close_kernel_size": 7,
    #     "min_contour_area": 2000     # Smaller area to include smaller tissue fragments
    # }
    # print("\nUsing custom parameters for segmentation.")
    # process_all_slides(params=custom_params, max_slides=2)

    # By default, it runs with the default parameters on a small subset of slides.
    print("\nUsing default parameters for segmentation.")
    process_all_slides(max_slides=2)

    print("\n--- Standalone script execution finished. ---")