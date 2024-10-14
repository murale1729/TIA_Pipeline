#!/usr/bin/env python

import os
import argparse
import cv2  # OpenCV for handling regular images
from tiatoolbox.wsicore.wsireader import WSIReader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Argument parser
parser = argparse.ArgumentParser(description="Tissue Masking for WSIs or Regular Images")
parser.add_argument('--input', type=str, help='Path to WSI or regular image file', required=True)
parser.add_argument('--output', type=str, help='Directory to save tissue mask', required=True)
parser.add_argument('--resolution', type=float, default=1.25, help='Resolution for tissue mask generation')
parser.add_argument('--units', type=str, choices=['mpp', 'power', 'level', 'baseline'], default='mpp', help='Units for resolution')
parser.add_argument('--mpp', type=float, help="Manually provide Microns Per Pixel (MPP) if missing in metadata", default=0.5)

args = parser.parse_args()

# Extract input filename (without extension)
input_filename = os.path.basename(args.input).split('.')[0]
output_dir = args.output

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Check if the input file exists
if not os.path.exists(args.input):
    raise FileNotFoundError(f"Input file {args.input} not found.")

# Handle both WSI files and regular images
if args.input.lower().endswith(('.svs', '.tiff', '.ndpi', '.vms')):
    # Handle WSIs
    wsi = WSIReader.open(args.input)
    
    # Extract metadata
    metadata = wsi.info.as_dict()
    mpp = metadata.get('mpp', (args.mpp, args.mpp))  # Use provided MPP if missing
    
    if not mpp:
        print(f"Warning: MPP not found in metadata, using default MPP of {args.mpp}")
        mpp = (args.mpp, args.mpp)
    
    # Generate tissue mask at the specified resolution and units
    mask = wsi.tissue_mask(resolution=args.resolution, units=args.units)
    mask_thumb = mask  # Use the mask directly

elif args.input.lower().endswith(('.png', '.jpg', '.jpeg')):
    # Handle regular images using OpenCV
    img = cv2.imread(args.input)
    if img is None:
        raise ValueError(f"Failed to load image: {args.input}")

    # Convert to grayscale and apply Otsu's threshold to create a binary mask
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask_thumb = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Dummy metadata (since regular images do not contain MPP)
    mpp = (args.mpp, args.mpp)

else:
    raise ValueError(f"Unsupported file format: {args.input}")

# Ensure the mask is writable if needed (in case it's read-only)
if not mask_thumb.flags.writeable:
    mask_thumb = np.copy(mask_thumb)
    mask_thumb.flags.writeable = True

# Convert the mask to a PIL Image and save as PNG
mask_image_pil = Image.fromarray(mask_thumb.astype(np.uint8))
mask_filename = f"{input_filename}_tissue_mask.png"
mask_path = os.path.join(output_dir, mask_filename)

# Save the tissue mask as a PNG image
mask_image_pil.save(mask_path)

# Optionally, visualize the results (useful for debugging)
plt.figure(figsize=(10, 5))
plt.imshow(mask_thumb, cmap='gray')
plt.title("Tissue Mask")
plt.axis("off")

# Save the visualization
visualization_path = os.path.join(output_dir, f"{input_filename}_mask_visualization.png")
plt.savefig(visualization_path)
# plt.show()  # Comment out if running in non-interactive environments

print(f"Tissue mask saved to: {mask_path}")
print(f"Mask visualization saved to: {visualization_path}")
