"""
This module implements WSI (Whole Slide Image) tissue segmentation
based on an adaptation of the CLAM algorithm, optimized to work
exclusively with the lowest-resolution image (thumbnail).

The process follows an efficient strategy:
1. The lowest-resolution version of the image (the highest-level thumbnail)
   is read from the WSI to ensure maximum speed.
2. Segmentation is performed on this small image using Otsu's method for
   automatic thresholding, making it more robust.
3. The resulting artifacts, such as the low-resolution mask, a visualization,
   and the tissue contours in GeoJSON format, are saved.
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
        - list: A list of the filtered contour arrays.
        - int: The WSI level used for processing.
    """
    try:
        wsi = openslide.OpenSlide(wsi_path)

        # --- Step 1: Work with the lowest-resolution image ---
        level = wsi.level_count - 1
        print(f"Using the lowest available resolution level: {level} for efficient segmentation.")

        thumb = wsi.read_region((0, 0), level, wsi.level_dimensions[level])
        # The image is read as a Pillow object and converted to a NumPy array for processing.
        # The shape will be (height, width, channels).
        thumb_rgb = np.array(thumb.convert('RGB'))
        wsi.close() # Close the WSI object as it's not needed for the rest of this function

        # --- Step 2: Segmentation on the low-resolution image ---
        hsv_image = cv2.cvtColor(thumb_rgb, cv2.COLOR_RGB2HSV)
        # We select the second channel (index 1) of the HSV image, which is Saturation.
        # This is done using NumPy's array slicing convention.
        saturation_channel = hsv_image[:, :, 1]

        blur_size = params.get("median_blur_size", DEFAULT_PARAMS["median_blur_size"])
        blurred_saturation = cv2.medianBlur(saturation_channel, blur_size)

        _, binary_mask = cv2.threshold(
            blurred_saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        close_kernel_size = params.get("close_kernel_size", DEFAULT_PARAMS["close_kernel_size"])
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size))
        closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

        min_area = params.get("min_contour_area", DEFAULT_PARAMS["min_contour_area"])
        contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

        # Create an empty (all zeros) NumPy array with the same dimensions as the mask.
        low_res_mask = np.zeros_like(closed_mask)
        cv2.drawContours(low_res_mask, filtered_contours, -1, (255), -1)

        return low_res_mask, thumb_rgb, filtered_contours, level

    except openslide.OpenSlideError as e:
        print(f"Error opening WSI file {wsi_path}: {e}")
        return None, None, None, None
    except Exception as e:
        print(f"An unexpected error occurred while processing {wsi_path}: {e}")
        return None, None, None, None

def save_contours_as_geojson(contours, wsi_path, level, output_path):
    """
    Saves detected contours in GeoJSON format, scaling them to the full WSI resolution.

    Args:
        contours (list): List of contour arrays from OpenCV.
        wsi_path (str): Path to the WSI file to get slide properties.
        level (int): The WSI level from which contours were extracted.
        output_path (str): The path to save the .geojson file.
    """
    try:
        wsi = openslide.OpenSlide(wsi_path)
        downsample_factor = wsi.level_downsamples[level]
        wsi.close()
    except openslide.OpenSlideError as e:
        print(f"Could not open {wsi_path} to get downsample factor: {e}")
        return

    features = []
    for contour in contours:
        # Scale contour from thumbnail coordinates to level 0 coordinates.
        # This is a vectorized operation in NumPy, multiplying each coordinate by the factor.
        scaled_contour = (contour * downsample_factor).astype(int)

        # Format for GeoJSON: needs to be a list of lists, and closed
        # .squeeze() removes redundant dimensions from the NumPy array.
        coordinates = scaled_contour.squeeze().tolist()

        # GeoJSON Polygons must be closed, so the first and last points must be the same.
        if coordinates and coordinates[0] != coordinates[-1]:
            coordinates.append(coordinates[0])

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [coordinates]  # GeoJSON requires an extra list wrapper for polygons
            },
            "properties": {}
        }
        features.append(feature)

    feature_collection = {
        "type": "FeatureCollection",
        "features": features
    }

    with open(output_path, 'w') as f:
        json.dump(feature_collection, f, indent=4)
    print(f"Contours saved in GeoJSON format to: {output_path}")


def visualize_segmentation(wsi_path, low_res_mask, thumb_rgb, level, output_path, save_figure=False, show_figure=False):
    """
    Generates and optionally saves or shows the low-resolution segmentation result.

    Args:
        wsi_path (str): Path to the WSI file.
        low_res_mask (np.ndarray): The binary segmentation mask.
        thumb_rgb (np.ndarray): The thumbnail image.
        level (int): The WSI level used.
        output_path (str): Path to save the figure.
        save_figure (bool): If True, saves the figure to disk.
        show_figure (bool): If True, displays the figure in a window.
    """
    # Create a copy of the thumbnail to draw on, preserving the original array.
    masked_thumb = thumb_rgb.copy()
    # Use boolean indexing from NumPy to set pixels to 0 where the mask is 0.
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

    if save_figure:
        plt.savefig(output_path)
        print(f"Visualization saved to: {output_path}")

    if show_figure:
        print("Displaying plot for the first slide...")
        plt.show()

    plt.close(fig)

def process_all_slides(params=DEFAULT_PARAMS, max_slides=None, save_visualization=True, show_first_slide=True):
    """
    Processes all WSI .tif files and saves the segmentation artifacts.

    Args:
        params (dict): Parameters for the segmentation algorithm.
        max_slides (int, optional): Maximum number of slides to process.
        save_visualization (bool): If True, saves the visualization PNG files.
        show_first_slide (bool): If True, shows the plot for the first slide processed.
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

    for i, wsi_path in enumerate(wsi_files):
        print(f"\n--- Processing: {os.path.basename(wsi_path)} ---")

        low_res_mask, thumb_rgb, contours, level = create_tissue_mask(wsi_path, params)

        if low_res_mask is not None:
            slide_name = os.path.splitext(os.path.basename(wsi_path))[0]
            slide_output_dir = os.path.join(output_base_dir, slide_name)
            os.makedirs(slide_output_dir, exist_ok=True)

            # --- Define output paths ---
            vis_path = os.path.join(slide_output_dir, f"{slide_name}_visualization.png")
            low_mask_path = os.path.join(slide_output_dir, f"{slide_name}_low_res_mask.png")
            params_path = os.path.join(slide_output_dir, f"{slide_name}_params.json")
            geojson_path = os.path.join(slide_output_dir, f"{slide_name}_contours.geojson")

            # --- Save artifacts ---
            # Determine if the plot should be shown for this specific slide
            should_show_plot = (i == 0 and show_first_slide)

            # Generate the visualization if it needs to be saved or shown
            if save_visualization or should_show_plot:
                visualize_segmentation(
                    wsi_path, low_res_mask, thumb_rgb, level, vis_path,
                    save_figure=save_visualization,
                    show_figure=should_show_plot
                )

            cv2.imwrite(low_mask_path, low_res_mask)
            print(f"Low-resolution mask saved to: {low_mask_path}")

            if contours:
                save_contours_as_geojson(contours, wsi_path, level, geojson_path)

            params_to_save = params.copy()
            params_to_save['level_used'] = level
            with open(params_path, 'w') as f:
                json.dump(params_to_save, f, indent=4)
            print(f"Parameters saved to: {params_path}")
        else:
            print(f"Skipping artifact generation for {os.path.basename(wsi_path)} due to a processing error.")

    print("\n--- All slides have been processed. ---")

if __name__ == '__main__':
    # This block allows the script to be run directly from the command line.
    # It's useful for testing the segmentation on a small number of slides.
    print("--- Running segmentation script in standalone mode ---")

    # By default, this will save the visualization and show the plot for the first slide.
    process_all_slides(max_slides=2, save_visualization=True, show_first_slide=True)

    print("\n--- Standalone script execution finished. ---")
