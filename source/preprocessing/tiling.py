"""
tiling.py

This module calculates the coordinates of tissue tiles from a WSI, using
previously generated segmentation contours from a GeoJSON file.

It now includes an integrated visualization function to display the final
tiling grid on the first processed WSI, removing the need for a separate
visualization script.
"""
import os
import glob
import openslide
import geopandas as gpd
from shapely.geometry import box
from shapely.affinity import scale
from PIL import Image
import cv2
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import csv
import random
import numpy as np

from source.config import locations
from source.config import config


def visualize_grid_on_wsi(wsi, wsi_path, geojson_path, valid_coords, params, output_path, show_figure, thumbnail_level=3):
    """
    Generates a high-resolution thumbnail and overlays the tile locations
    as a proper grid of rectangles. This function replaces the old visualizer.
    """
    print("  Generating detailed grid visualization...")
    try:
        if not valid_coords:
            print("  Skipping visualization: No valid coordinates found.")
            return

        tissue_polygons_gdf = gpd.read_file(geojson_path)

        # Get thumbnail and scaling info
        downsample_factor = wsi.level_downsamples[thumbnail_level]
        thumbnail_dims = wsi.level_dimensions[thumbnail_level]
        thumbnail_img = wsi.read_region((0, 0), thumbnail_level, thumbnail_dims).convert("RGB")

        # Create the plot
        fig, ax = plt.subplots(figsize=(20, 20))
        ax.imshow(thumbnail_img)
        ax.set_title(f'Tiling Grid for {os.path.basename(wsi_path)} | Level: {thumbnail_level} | Tiles: {len(valid_coords)}')

        # Draw tissue polygons (blue line)
        scaled_polygons = [scale(geom, xfact=1/downsample_factor, yfact=1/downsample_factor, origin=(0,0))
                           for geom in tissue_polygons_gdf.geometry]
        gpd.GeoDataFrame(geometry=scaled_polygons).plot(ax=ax, edgecolor='blue', facecolor='none', linewidth=2)

        # Draw the actual grid using PatchCollection
        # The base magnification is now read from the config file.
        base_mag = config.WSI_BASE_MAGNIFICATION
        patch_size_at_base_mag = int(params["patch_size"] * (base_mag / params["magnification"]))
        scaled_patch_size = patch_size_at_base_mag / downsample_factor

        rects = []
        for x, y in valid_coords:
            rect = patches.Rectangle(
                (x / downsample_factor, y / downsample_factor),
                scaled_patch_size,
                scaled_patch_size
            )
            rects.append(rect)

        # Create a single collection of all tile patches
        tile_collection = PatchCollection(
            rects,
            edgecolor='lime',   # Color of the grid lines
            linewidth=0.5,      # Thin lines for a tight grid
            facecolor='lime',   # Fill color for the squares
            alpha=0.25          # Semi-transparent fill to see tissue and grid
        )
        ax.add_collection(tile_collection)

        plt.axis('off')
        plt.tight_layout()

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path)
            print(f"  Visualization saved to: {output_path}")

        if show_figure:
            print("  Displaying plot for the first slide...")
            plt.show()

        plt.close(fig)

    except Exception as e:
        print(f"  An error occurred during detailed visualization: {e}")


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

    # The base magnification is now read from the config file.
    base_mag = config.WSI_BASE_MAGNIFICATION

    patch_size_level0 = int(patch_size_at_target_mag * (base_mag / target_mag))
    step_size_level0 = patch_size_level0

    width, height = wsi.dimensions
    valid_coords = []

    print(f"  Calculating valid coordinates with a forced step size of {step_size_level0}px...")
    for y in range(0, height - patch_size_level0, step_size_level0):
        for x in range(0, width - patch_size_level0, step_size_level0):
            patch_box = box(x, y, x + patch_size_level0, y + patch_size_level0)
            if patch_box.intersects(tissue_unary):
                if (patch_box.intersection(tissue_unary).area / patch_box.area) > params["tissue_threshold"]:
                    valid_coords.append((x, y))

    if visualize:
        # Call the new, integrated visualization function
        visualize_grid_on_wsi(wsi, wsi_path, geojson_path, valid_coords, params, vis_output_path, show_figure)

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
    tiling_base_dir = locations.get_tiling_output_dir()
    visualizations_base_dir = locations.get_visualizations_output_dir()

    segmentation_dirs = sorted([d for d in glob.glob(os.path.join(segmentation_base_dir, "*")) if os.path.isdir(d)])

    if not segmentation_dirs:
        print(f"No segmentation directories found in: {segmentation_base_dir}")
        return

    if max_slides is not None:
        segmentation_dirs = segmentation_dirs[:max_slides]
        print(f"Processing a subset of {len(segmentation_dirs)} slides.")

    for i, seg_slide_dir in enumerate(segmentation_dirs):
        slide_name = os.path.basename(seg_slide_dir)
        print(f"\n--- Processing: {slide_name} ---")

        wsi_path = os.path.join(wsi_dir, f"{slide_name}.tif")
        geojson_path = os.path.join(seg_slide_dir, f"{slide_name}_contours.geojson")

        tiling_output_slide_dir = os.path.join(tiling_base_dir, slide_name)
        coords_output_path = os.path.join(tiling_output_slide_dir, "coordinates.csv")
        
        visualizations_slide_dir = os.path.join(visualizations_base_dir, slide_name)
        vis_output_path = os.path.join(visualizations_slide_dir, f"{slide_name}_tiling_grid_visualization.png")

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
