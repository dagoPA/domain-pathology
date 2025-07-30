"""
CONCH v1.5 Feature Extraction for Pathology Images
Extracts patch-level features from WSI images using the local conchv1_5.py model definition.
"""

import os
import glob
import h5py
import numpy as np
import torch
from PIL import Image
import openslide
import cv2
import logging
from tqdm import tqdm
from torch.utils.data import Dataset
import torchvision.transforms as T

from source.features.conchv1_5 import create_model_from_pretrained
from source.config.config import (PATCH_SIZE, USE_TISSUE_SEGMENTATION,
                                STHRESH, STHRESH_UP, MTHRESH, CLOSE, USE_OTSU,
                                FILTER_PARAMS, REF_PATCH_SIZE, EXCLUDE_IDS)

from source.config.locations import get_dataset_dir, get_features_output_dir, get_conch_model_path

# Define OpenAI CLIP normalization constants, as they are used in the preprocessing pipeline
OPENAI_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_STD = (0.26862954, 0.26130258, 0.27577711)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PatchDataset(Dataset):
    """Dataset for processing WSI patches efficiently using OpenSlide with tissue segmentation."""

    def __init__(self, image_path, patch_size=PATCH_SIZE, level=0):
        self.image_path = image_path
        self.patch_size = patch_size
        self.level = level
        self.use_tissue_segmentation = USE_TISSUE_SEGMENTATION
        self.sthresh = STHRESH
        self.sthresh_up = STHRESH_UP
        self.mthresh = MTHRESH
        self.close = CLOSE
        self.use_otsu = USE_OTSU
        self.filter_params = FILTER_PARAMS
        self.ref_patch_size = REF_PATCH_SIZE
        self.exclude_ids = EXCLUDE_IDS
        self.slide = None
        self.seg_level = -1  # Placeholder, will be set dynamically in _initialize_slide
        self.patch_coords = []
        self.contours_tissue = []
        self.holes_tissue = []
        self._initialize_slide()

    def _filter_contours(self, contours, hierarchy, filter_params):
        filtered = []
        hierarchy_1 = np.flatnonzero(hierarchy[:, 1] == -1)
        all_holes = []
        for cont_idx in hierarchy_1:
            cont = contours[cont_idx]
            holes = np.flatnonzero(hierarchy[:, 1] == cont_idx)
            a = cv2.contourArea(cont)
            hole_areas = [cv2.contourArea(contours[hole_idx]) for hole_idx in holes]
            a = a - np.array(hole_areas).sum()
            if a == 0:
                continue
            if tuple((filter_params['a_t'],)) < tuple((a,)):
                filtered.append(cont_idx)
                all_holes.append(holes)
        foreground_contours = [contours[cont_idx] for cont_idx in filtered]
        hole_contours = []
        for hole_ids in all_holes:
            unfiltered_holes = [contours[idx] for idx in hole_ids]
            unfiltered_holes = sorted(unfiltered_holes, key=cv2.contourArea, reverse=True)
            unfiltered_holes = unfiltered_holes[:filter_params['max_n_holes']]
            filtered_holes = []
            for hole in unfiltered_holes:
                if cv2.contourArea(hole) > filter_params['a_h']:
                    filtered_holes.append(hole)
            hole_contours.append(filtered_holes)
        return foreground_contours, hole_contours

    def _scale_contour_dim(self, contours, scale):
        return [np.array(cont * scale, dtype='int32') for cont in contours]

    def _scale_holes_dim(self, hole_contours, scale):
        return [[np.array(hole * scale, dtype='int32') for hole in holes] for holes in hole_contours]

    def _segment_tissue(self):
        if not self.slide:
            return
        img = np.array(self.slide.read_region((0, 0), self.seg_level, self.slide.level_dimensions[self.seg_level]))
        img = img[:, :, :3]
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        img_med = cv2.medianBlur(img_hsv[:, :, 1], self.mthresh)
        if self.use_otsu:
            _, img_otsu = cv2.threshold(img_med, 0, self.sthresh_up, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
        else:
            _, img_otsu = cv2.threshold(img_med, self.sthresh, self.sthresh_up, cv2.THRESH_BINARY)
        if self.close > 0:
            kernel = np.ones((self.close, self.close), np.uint8)
            img_otsu = cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, kernel)
        scale = self.slide.level_downsamples[self.seg_level]
        scaled_ref_patch_area = int(self.ref_patch_size**2 / (scale * scale))
        filter_params = self.filter_params.copy()
        filter_params['a_t'] *= scaled_ref_patch_area
        filter_params['a_h'] *= scaled_ref_patch_area
        contours, hierarchy = cv2.findContours(img_otsu, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]
        if len(contours) > 0 and filter_params:
            foreground_contours, hole_contours = self._filter_contours(contours, hierarchy, filter_params)
            self.contours_tissue = self._scale_contour_dim(foreground_contours, scale)
            self.holes_tissue = self._scale_holes_dim(hole_contours, scale)
            contour_ids = set(np.arange(len(self.contours_tissue))) - set(self.exclude_ids)
            self.contours_tissue = [self.contours_tissue[i] for i in contour_ids if i < len(self.contours_tissue)]
            self.holes_tissue = [self.holes_tissue[i] for i in contour_ids if i < len(self.holes_tissue)]

    def _is_in_tissue(self, x, y, patch_size):
        if not self.contours_tissue:
            return True
        center_x, center_y = x + patch_size // 2, y + patch_size // 2
        for i, contour in enumerate(self.contours_tissue):
            if cv2.pointPolygonTest(contour, (center_x, center_y), False) >= 0:
                if i < len(self.holes_tissue):
                    for hole in self.holes_tissue[i]:
                        if cv2.pointPolygonTest(hole, (center_x, center_y), False) >= 0:
                            return False
                return True
        return False

    def _initialize_slide(self):
        try:
            self.slide = openslide.OpenSlide(self.image_path)

            # Dynamically determine the segmentation level.
            # Use the second to last level (penultimate) for segmentation, as it provides a good
            # balance between tissue overview and detail, improving performance.
            if self.slide.level_count > 1:
                self.seg_level = self.slide.level_count - 3
            else:
                # Fallback for slides with only one level.
                self.seg_level = self.slide.level_count - 1
            logger.info(f"Using segmentation level: {self.seg_level} (from {self.slide.level_count} available levels)")

            if self.use_tissue_segmentation:
                logger.info("Performing tissue segmentation...")
                self._segment_tissue()
                logger.info(f"Found {len(self.contours_tissue)} tissue regions")

            level_dims = self.slide.level_dimensions[self.level]
            width, height = level_dims
            all_coords = []
            for y in range(0, height - self.patch_size + 1, self.patch_size):
                for x in range(0, width - self.patch_size + 1, self.patch_size):
                    downsample = self.slide.level_downsamples[self.level]
                    level0_x = int(x * downsample)
                    level0_y = int(y * downsample)
                    all_coords.append((level0_x, level0_y))

            if self.use_tissue_segmentation and self.contours_tissue:
                logger.info(f"Filtering {len(all_coords)} patches based on tissue segmentation...")
                for coord in all_coords:
                    if self._is_in_tissue(coord[0], coord[1], int(self.patch_size * self.slide.level_downsamples[self.level])):
                        self.patch_coords.append(coord)
                logger.info(f"Kept {len(self.patch_coords)} patches in tissue regions")
            else:
                self.patch_coords = all_coords
        except Exception as e:
            logger.error(f"Error initializing slide {self.image_path}: {e}")
            try:
                image = Image.open(self.image_path).convert('RGB')
                width, height = image.size
                for y in range(0, height - self.patch_size + 1, self.patch_size):
                    for x in range(0, width - self.patch_size + 1, self.patch_size):
                        self.patch_coords.append((x, y))
                self._use_pil_fallback = True
                self._pil_image = image
                logger.warning("Using PIL fallback - tissue segmentation disabled")
            except Exception as e2:
                logger.error(f"PIL fallback also failed: {e2}")
                self.patch_coords = []

    def __len__(self):
        return len(self.patch_coords)

    def __getitem__(self, idx):
        if idx >= len(self.patch_coords):
            raise IndexError("Patch index out of range")
        x, y = self.patch_coords[idx]
        try:
            if hasattr(self, '_use_pil_fallback') and self._use_pil_fallback:
                patch = self._pil_image.crop((x, y, x + self.patch_size, y + self.patch_size))
            else:
                patch = self.slide.read_region((x, y), self.level, (self.patch_size, self.patch_size)).convert('RGB')
            return patch
        except Exception as e:
            logger.warning(f"Error extracting patch at ({x}, {y}): {e}")
            return Image.new('RGB', (self.patch_size, self.patch_size), (255, 255, 255))

    def __del__(self):
        if self.slide:
            self.slide.close()


class CONCHFeatureExtractor:
    """
    Feature extractor for CONCH v1.5, using the model definition
    from the local `conchv1_5.py` file.
    """

    def __init__(self, device=None, img_size=448):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.preprocess = None
        self.img_size = img_size
        self._load_model()

    def _load_model(self):
        """Loads the CONCH model and preprocessor using the local script."""
        try:
            model_path = get_conch_model_path()
            logger.info(f"Loading model from local path: {model_path}...")

            self.model, _ = create_model_from_pretrained(
                checkpoint_path=model_path,
                img_size=self.img_size
            )

            self.preprocess = T.Compose([
                T.Resize(self.img_size, interpolation=T.InterpolationMode.BILINEAR),
                T.CenterCrop(self.img_size),
                T.ToTensor(),
                T.Normalize(mean=OPENAI_MEAN, std=OPENAI_STD)
            ])

            self.model.to(self.device)
            self.model.eval()
            logger.info(f"CONCH v1.5 model loaded successfully from {model_path} on {self.device}")
            logger.info("Preprocessing pipeline configured with custom OpenAI constants.")

        except Exception as e:
            logger.error(f"Error loading CONCH v1.5 model from local script: {e}")
            raise

    def extract_features_from_dataset(self, dataset, batch_size=32):
        """Extracts features from a PatchDataset efficiently."""
        all_features = []
        with torch.no_grad():
            for i in tqdm(range(0, len(dataset), batch_size), desc="Extracting Features"):
                batch_patches_pil = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]

                if not batch_patches_pil:
                    continue

                batch_tensors = [self.preprocess(patch) for patch in batch_patches_pil]
                batch_input = torch.stack(batch_tensors).to(self.device)

                batch_features = self.model(batch_input)
                all_features.append(batch_features.cpu().numpy())

        return np.vstack(all_features) if all_features else np.array([])

    def process_slide(self, slide_path, output_path, patch_size=PATCH_SIZE):
        """Processes a single slide and saves features."""
        if os.path.exists(output_path):
            logger.info(f"Skipping existing file: {output_path}")
            return

        logger.info(f"Processing slide: {slide_path}")
        dataset = PatchDataset(slide_path, patch_size=patch_size)

        if len(dataset) == 0:
            logger.warning(f"No patches extracted from {slide_path}")
            return

        logger.info(f"Extracting features from {len(dataset)} patches...")
        features = self.extract_features_from_dataset(dataset)

        if len(features) == 0:
            logger.warning(f"No features extracted from {slide_path}")
            return

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('features', data=features)
            f.attrs['slide_path'] = slide_path
            f.attrs['patch_size'] = patch_size
            f.attrs['num_patches'] = len(features)
        logger.info(f"Saved {len(features)} patch features to {output_path}")


def extract_features_from_dataset():
    """Main function to extract features from all slides in the dataset."""
    extractor = CONCHFeatureExtractor()

    dataset_dir = get_dataset_dir()

    image_extensions = ['*.tif', '*.tiff']
    image_files = []

    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(dataset_dir, '**', ext), recursive=True))

    logger.info(f"Found {len(image_files)} image files to process")

    for slide_path in tqdm(image_files, desc="Processing slides"):
        slide_name = os.path.splitext(os.path.basename(slide_path))[0]
        output_path = os.path.join(get_features_output_dir(), f"{slide_name}.h5")

        try:
            extractor.process_slide(slide_path, output_path)
        except Exception as e:
            logger.error(f"Error processing {slide_path}: {e}")
            continue

    logger.info("Feature extraction completed!")


if __name__ == "__main__":
    extract_features_from_dataset()
