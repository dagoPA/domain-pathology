"""
Simple visualization module for CAMELYON17 WSI thumbnails.
Plots thumbnail of the first image found in the directory.
Uses OpenSlide for WSI file handling.
"""

import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import glob
import openslide
# Import configuration
from source.config.locations import get_dataset_dir

Image.MAX_IMAGE_PIXELS = None


def visualize_wsi_thumbnail():
    """Find and visualize thumbnail of the first WSI file found in the dataset."""
    print("Starting simple CAMELYON17 WSI thumbnail visualization...")

    # Get dataset directory and search for WSI files
    dataset_dir = get_dataset_dir()
    print(f"Searching for WSI files in: {dataset_dir}")

    # Search for files with supported extensions (only TIF files as requested)
    wsi_extensions = ['.tif', '.tiff']
    first_file = None

    for ext in wsi_extensions:
        pattern = os.path.join(dataset_dir, f"**/*{ext}")
        files = glob.glob(pattern, recursive=True)
        if files:
            first_file = files[0]
            print(f"Found first WSI file: {first_file}")
            break

    if not first_file:
        print("No WSI files found to plot.")
        return

    # Create thumbnail
    print(f"Creating thumbnail for: {os.path.basename(first_file)}")
    size = (600, 600)

    try:
        # Use OpenSlide to open WSI files
        slide = openslide.OpenSlide(first_file)

        # Get the best level for thumbnail (usually the lowest resolution level)
        # OpenSlide provides multiple levels, with level 0 being the highest resolution
        best_level = slide.level_count - 1

        # Get thumbnail from the slide
        # OpenSlide's get_thumbnail() method automatically handles the conversion
        thumbnail_pil = slide.get_thumbnail(size)

        # Convert to RGB if necessary
        if thumbnail_pil.mode != 'RGB':
            thumbnail_pil = thumbnail_pil.convert('RGB')

        # Convert to numpy array for matplotlib
        thumbnail = np.array(thumbnail_pil)

        # Close the slide
        slide.close()

    except Exception as e:
        print(f"Error creating thumbnail with OpenSlide for {first_file}: {e}")
        print("Trying fallback with PIL...")
        try:
            # Fallback to PIL for regular TIFF files that might not be WSI format
            with Image.open(first_file) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.thumbnail(size, Image.Resampling.LANCZOS)
                thumbnail = np.array(img)
        except Exception as e2:
            print(f"Fallback also failed: {e2}")
            # Return a placeholder image
            thumbnail = np.ones((*size, 3), dtype=np.uint8) * 128  # Gray placeholder

    # Create and show plot with proper background configuration for Jupyter
    fig, ax = plt.subplots(figsize=(8, 8))

    # Set white background to avoid transparency issues in Jupyter dark themes
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    ax.imshow(thumbnail)
    ax.set_title(f'WSI Thumbnail: {os.path.basename(first_file)}', fontsize=14, color='black')
    ax.axis('off')
    plt.tight_layout()
    plt.show()

    print("Visualization complete!")


if __name__ == "__main__":
    visualize_wsi_thumbnail()
