import os
import argparse
import cv2  # OpenCV for handling regular images
from tiatoolbox.tools.tissuemask import MorphologicalMasker
import matplotlib.pyplot as plt
import numpy as np

# Argument parser
parser = argparse.ArgumentParser(description="Tissue Masking for WSIs or Regular Images")
parser.add_argument('--input', type=str, help='Path to WSI or regular image file', required=True)
parser.add_argument('--output', type=str, help='Directory to save tissue mask', required=True)
parser.add_argument('--resolution', type=float, default=1.25, help='Resolution for tissue mask generation (only used for WSIs)')

args = parser.parse_args()

# Extract input filename (without extension) to create output folder, if needed
input_filename = os.path.basename(args.input).split('.')[0]
output_dir = args.output

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Check if the input file exists
if not os.path.exists(args.input):
    raise FileNotFoundError(f"Input file {args.input} not found.")

# Try to handle both WSI files and regular images
if args.input.lower().endswith(('.svs', '.tiff', '.ndpi', '.vms')):
    # Handle WSIs
    from tiatoolbox.wsicore.wsireader import WSIReader
    wsi = WSIReader.open(args.input)
    mask = wsi.tissue_mask(resolution=args.resolution, units="power")
    mask_thumb = mask.slide_thumbnail(resolution=args.resolution, units="power")

elif args.input.lower().endswith(('.png', '.jpg', '.jpeg')):
    # Handle regular images using OpenCV
    img = cv2.imread(args.input)
    if img is None:
        raise ValueError(f"Failed to load image: {args.input}")

    # Convert to grayscale and apply Otsu's threshold to create a binary mask
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask_thumb = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

else:
    raise ValueError(f"Unsupported file format: {args.input}")

# Ensure the mask is writable if needed (in case it's read-only)
if not mask_thumb.flags.writeable:
    mask_thumb = np.copy(mask_thumb)
    mask_thumb.flags.writeable = True

# Generate the full path for the output mask file (ensure it's not treated as a directory)
mask_filename = f"{input_filename}_tissue_mask.png"
mask_path = os.path.join(output_dir, mask_filename)

# Save the tissue mask as an image
cv2.imwrite(mask_path, mask_thumb)

# Optionally, visualize the results (useful for debugging)
plt.figure(figsize=(10, 5))
plt.imshow(mask_thumb, cmap='gray')
plt.title("Tissue Mask")
plt.axis("off")

# Save the visualization
visualization_path = os.path.join(output_dir, f"{input_filename}_mask_visualization.png")
plt.savefig(visualization_path)
plt.show()

print(f"Tissue mask saved to: {mask_path}")
print(f"Mask visualization saved to: {visualization_path}")
