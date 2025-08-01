"""
tiling.py

This module calculates the coordinates of tissue tiles from a WSI, using
previously generated segmentation contours from a GeoJSON file.
"""
import os
import glob
import openslide
import geopandas as gpd
from shapely.geometry import box
from PIL import Image
import cv2
from matplotlib import pyplot as plt
import csv
import random

from source.config import locations
from source.config import config


def visualize_tiling(wsi, wsi_path, valid_coords, params, output_path, show_figure):
    """
    Generates a 2x5 grid visualization of 10 random sample tiles.
    """
    print("  Generating visualization...")
    try:
        if not valid_coords:
            print("  Skipping visualization: No valid coordinates found.")
            return

        # --- Setup 2x5 Subplot Grid ---
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        fig.suptitle(f"Sample Tiles for {os.path.basename(wsi_path)}", fontsize=16)

        # --- Get Sample Tiles ---
        sample_coords = random.sample(valid_coords, min(10, len(valid_coords)))

        try:
            base_mag = float(wsi.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
        except (KeyError, ValueError):
            base_mag = 40.0

        patch_size_at_base_mag = int(params["patch_size"] * (base_mag / params["magnification"]))

        # --- Draw Sample Tiles in Subplots ---
        ax_flat = axes.flatten()
        for i, (x, y) in enumerate(sample_coords):
            tile_pil = wsi.read_region((x, y), 0, (patch_size_at_base_mag, patch_size_at_base_mag)).convert("RGB")
            # Resize to final patch size for consistent display
            tile_resized = tile_pil.resize((params["patch_size"], params["patch_size"]), Image.Resampling.LANCZOS)

            ax_flat[i].imshow(tile_resized)
            ax_flat[i].set_title(f"({x},{y})", fontsize=8)
            ax_flat[i].axis('off')

        # Turn off axes for any unused subplots
        for i in range(len(sample_coords), 10):
            ax_flat[i].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(output_path)
        print(f"  Visualization saved to: {output_path}")

        if show_figure:
            print("  Displaying plot for the first slide...")
            plt.show()

        plt.close(fig)

    except Exception as e:
        print(f"  An error occurred during visualization: {e}")


def calculate_and_save_coords(wsi_path, geojson_path, output_csv_path, params, visualize=False, vis_output_path=None, show_figure=False):
    """
    Calculates tile coordinates and saves them to a CSV file.
    """
    print(f"  Processing WSI: {os.path.basename(wsi_path)}")
    try:
        wsi = openslide.OpenSlide(wsi_path)
        tissue_contours = gpd.read_file(geojson_path)
    except Exception as e:
        print(f"  Error opening files for {os.path.basename(wsi_path)}: {e}")
        return

    tissue_unary = tissue_contours.union_all()

    target_mag = params["magnification"]
    patch_size_at_target_mag = params["patch_size"]
    step_size = patch_size_at_target_mag - params["overlap"]

    try:
        base_mag = float(wsi.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
    except (KeyError, ValueError):
        print("  Warning: Could not read base magnification. Assuming 40x.")
        base_mag = 40.0

    patch_size_level0 = int(patch_size_at_target_mag * (base_mag / target_mag))
    step_size_level0 = int(step_size * (base_mag / target_mag))

    width, height = wsi.dimensions
    valid_coords = []

    print("  Calculating valid coordinates...")
    for y in range(0, height - patch_size_level0, step_size_level0):
        for x in range(0, width - patch_size_level0, step_size_level0):
            patch_box = box(x, y, x + patch_size_level0, y + patch_size_level0)
            if patch_box.intersects(tissue_unary):
                if (patch_box.intersection(tissue_unary).area / patch_box.area) > params["tissue_threshold"]:
                    valid_coords.append((x, y))

    if visualize:
        # Note: The 'tissue_unary' argument is no longer needed for the new visualization
        visualize_tiling(wsi, wsi_path, valid_coords, params, vis_output_path, show_figure)

    if not valid_coords:
        print("  No valid tissue patches were found.")
        wsi.close()
        return

    print(f"  Found {len(valid_coords)} valid coordinates. Saving to CSV...")
    try:
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        with open(output_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['x', 'y'])
            writer.writerows(valid_coords)
        print(f"  Coordinates saved to: {output_csv_path}")
    except Exception as e:
        print(f"  Error saving coordinates to CSV: {e}")

    wsi.close()


def process_all_slides(max_slides=None, show_first_slide_tiling=False):
    """
    Finds all segmentation results and runs the coordinate calculation process.
    """
    params = {
        "patch_size": config.TILE_PATCH_SIZE,
        "magnification": config.TILE_MAGNIFICATION,
        "overlap": config.TILE_OVERLAP,
        "tissue_threshold": config.TILE_TISSUE_THRESHOLD,
    }

    wsi_dir = locations.get_dataset_dir()
    segmentation_base_dir = locations.get_segmentation_output_dir()
    segmentation_dirs = sorted([d for d in glob.glob(os.path.join(segmentation_base_dir, "*")) if os.path.isdir(d)])

    if not segmentation_dirs:
        print(f"No segmentation directories found in: {segmentation_base_dir}")
        return

    if max_slides is not None:
        segmentation_dirs = segmentation_dirs[:max_slides]
        print(f"Processing a subset of {len(segmentation_dirs)} slides.")

    for i, slide_dir in enumerate(segmentation_dirs):
        slide_name = os.path.basename(slide_dir)
        print(f"\n--- Processing: {slide_name} ---")

        wsi_path = os.path.join(wsi_dir, f"{slide_name}.tif")
        geojson_path = os.path.join(slide_dir, f"{slide_name}_contours.geojson")
        coords_output_path = os.path.join(slide_dir, "coordinates.csv")
        vis_output_path = os.path.join(slide_dir, f"{slide_name}_tiling_visualization.png")

        if not (os.path.exists(wsi_path) and os.path.exists(geojson_path)):
            print(f"  Error: Missing WSI or GeoJSON file for {slide_name}.")
            continue

        should_visualize = (i == 0 and show_first_slide_tiling)

        calculate_and_save_coords(
            wsi_path=wsi_path,
            geojson_path=geojson_path,
            output_csv_path=coords_output_path,
            params=params,
            visualize=should_visualize,
            vis_output_path=vis_output_path,
            show_figure=should_visualize
        )

    print("\n--- All slides have been processed. ---")


if __name__ == '__main__':
    print("--- Running coordinate calculation script in standalone mode ---")
    process_all_slides(max_slides=2, show_first_slide_tiling=True)
    print("\n--- Standalone script execution finished. ---")