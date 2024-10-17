#!/usr/bin/env python 

import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import numpy as np
import pickle
from tiatoolbox import data, logger
from tiatoolbox.tools import stainnorm
from tiatoolbox.wsicore.wsireader import WSIReader

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Stain Normalization with PNG output and metadata storage")
parser.add_argument('--input', type=str, required=True, help='Path to input WSI file')
parser.add_argument('--output', type=str, required=True, help='Path to save normalized WSI image as PNG')
parser.add_argument('--reference', type=str, help='Path to reference image for stain normalization', default=None)
parser.add_argument('--method', type=str, choices=['vahadane', 'macenko', 'reinhard', 'ruifrok'], default='vahadane', help='Stain normalization method to use')

args = parser.parse_args()

# Set up logging
logger.setLevel('INFO')

# Load the WSI and extract metadata
wsi_reader = WSIReader.open(args.input)
metadata = wsi_reader.info.as_dict()  # Save metadata to reapply later

# Extract full-resolution WSI or appropriate resolution based on requirements
logger.info("Extracting slide image at the selected resolution...")
level = 0  # Select the highest resolution (level 0)
slide_image = wsi_reader.read_region(location=(0, 0), level=level, size=wsi_reader.slide_dimensions(level=level))

# Convert to NumPy array
slide_image_writable = np.array(slide_image)

# Ensure that the array is writable
if not slide_image_writable.flags.writeable:
    slide_image_writable = np.copy(slide_image_writable)  # Ensure it's writable

# Log the dimensions of the input image
logger.info(f"Slide image dimensions (H x W x C): {slide_image_writable.shape}")

# Load or set the reference image
if args.reference:
    reference_image = plt.imread(args.reference)
    logger.info(f"Using provided reference image: {args.reference}")
else:
    reference_image = data.stain_norm_target()
    logger.info("Using default reference image from tiatoolbox.")

# Initialize the stain normalizer
if args.method == 'vahadane':
    stain_normalizer = stainnorm.VahadaneNormalizer()
elif args.method == 'macenko':
    stain_normalizer = stainnorm.MacenkoNormalizer()
elif args.method == 'reinhard':
    stain_normalizer = stainnorm.ReinhardNormalizer()
elif args.method == 'ruifrok':
    stain_normalizer = stainnorm.RuifrokNormalizer()
else:
    raise ValueError(f"Unsupported stain normalization method: {args.method}")

# Fit the normalizer to the reference image
logger.info("Fitting the stain normalizer to the reference image...")
stain_normalizer.fit(reference_image)

# Perform stain normalization on the slide image
logger.info("Applying stain normalization to the slide image...")
normalized_image = stain_normalizer.transform(slide_image_writable)

# Ensure the output directory exists
output_path = Path(args.output)
output_dir = output_path.parent
output_dir.mkdir(parents=True, exist_ok=True)

# Check if normalized image has 4 channels (RGBA) and convert to RGB
if normalized_image.shape[-1] == 4:  # If RGBA, remove the alpha channel
    logger.info("Converting RGBA to RGB for PNG compatibility...")
    normalized_image = normalized_image[:, :, :3]

# Normalize the image values to [0, 255] for saving as PNG
normalized_image = (normalized_image * 255).clip(0, 255).astype(np.uint8)

# Convert NumPy array to PIL Image for saving as PNG
normalized_image_pil = Image.fromarray(normalized_image)

# Save the normalized image as PNG
png_output_path = output_path.with_suffix('.png')
logger.info(f"Saving the normalized image to {png_output_path}...")
normalized_image_pil.save(png_output_path)

# Store the metadata for later use (e.g., segmentation)
metadata_path = output_dir / 'metadata.pkl'
logger.info(f"Saving metadata to {metadata_path}...")
with open(metadata_path, 'wb') as f:
    pickle.dump(metadata, f)

logger.info(f"Stain normalization completed. Normalized PNG image saved to {png_output_path}")
logger.info(f"Metadata saved to {metadata_path}")
