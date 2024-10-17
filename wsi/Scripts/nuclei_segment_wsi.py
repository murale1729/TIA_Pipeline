import argparse
import os
import logging
import torch
import numpy as np
import time
from math import ceil
from tiatoolbox.models.engine.nucleus_instance_segmentor import NucleusInstanceSegmentor
from tiatoolbox.wsicore.wsireader import WSIReader

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Parsing input arguments
parser = argparse.ArgumentParser(description="Nuclei Segmentation with Progress Tracking and Resume Capability")
parser.add_argument('--input', type=str, help='Path to normalized image or WSI', required=True)
parser.add_argument('--output_dir', type=str, help='Directory to save output results', required=True)
parser.add_argument('--gpu', action='store_true', help='Use GPU for processing')
parser.add_argument('--default_mpp', type=float, help="Default MPP if not found in metadata", default=0.5)
parser.add_argument('--batch_size', type=int, help='Batch size for processing', default=64)
parser.add_argument('--num_loader_workers', type=int, help='Number of data loader workers', default=16)
parser.add_argument('--num_postproc_workers', type=int, help='Number of post-processing workers', default=16)
args = parser.parse_args()

# Check for GPU usage
if not args.gpu:
    args.gpu = torch.cuda.is_available()

device = torch.device("cuda" if args.gpu else "cpu")
logger.info(f"Using device: {device}")

# Load the WSI and extract metadata
wsi_reader = WSIReader.open(args.input)
metadata = wsi_reader.info.as_dict()
logger.info(f"WSI Metadata: {metadata}")

# Extract MPP
mpp = metadata.get('mpp', None)
if isinstance(mpp, (tuple, list)) and len(mpp) == 2:
    mpp_value = sum(mpp) / len(mpp)
elif isinstance(mpp, (int, float)):
    mpp_value = float(mpp)
else:
    mpp_value = args.default_mpp
    logger.warning(f"MPP not found, using default MPP: {mpp_value}")

logger.info(f"Microns per pixel (MPP) used: {mpp_value}")

# Initialize NucleusInstanceSegmentor
logger.info("Initializing NucleusInstanceSegmentor")
segmentor = NucleusInstanceSegmentor(
    pretrained_model="hovernet_fast-pannuke",
    num_loader_workers=args.num_loader_workers,   # Adjust based on your system
    num_postproc_workers=args.num_postproc_workers, # Adjust based on your system
    batch_size=args.batch_size,           # Adjust based on your GPU memory
    auto_generate_mask=False
)
segmentor.model.to(device)

# Define tile parameters
tile_size = [5000, 5000]  # Adjust as needed
tile_overlap = [250, 250]  # Adjust as needed
stride = [tile_size[0] - tile_overlap[0], tile_size[1] - tile_overlap[1]]

# Calculate total tiles
wsi_dimensions = wsi_reader.slide_dimensions
total_tiles = 1
for dim, size, st in zip(wsi_dimensions, tile_size, stride):
    num_tiles = ceil((dim - size) / st) + 1
    total_tiles *= num_tiles
logger.info(f"Total tiles to process: {total_tiles}")

# Load processed tiles log
processed_tiles_log = os.path.join(args.output_dir, 'processed_tiles.log')

def log_processed_tile(tile_index):
    with open(processed_tiles_log, 'a') as f:
        f.write(f"{tile_index}\n")

def load_processed_tiles():
    if os.path.exists(processed_tiles_log):
        with open(processed_tiles_log, 'r') as f:
            processed_tiles = set(int(line.strip()) for line in f)
    else:
        processed_tiles = set()
    return processed_tiles

processed_tiles = load_processed_tiles()

# Generate tile coordinates
from tiatoolbox.utils.misc import get_tile_coords

tile_coords = get_tile_coords(
    wsi_dimensions=wsi_dimensions,
    tile_size=tile_size,
    stride=stride
)

# Start processing
start_time = time.time()
processed_count = len(processed_tiles)
for idx, coords in enumerate(tile_coords):
    if idx in processed_tiles:
        logger.info(f"Skipping tile {idx+1}, already processed.")
        continue

    logger.info(f"Processing tile {idx+1}/{total_tiles}")

    # Extract tile region
    x, y, w, h = coords
    tile_image = wsi_reader.read_region((x, y), level=0, size=(w, h))

    # Save tile image temporarily
    tile_image_path = os.path.join(args.output_dir, f"tile_{idx}.png")
    tile_image.save(tile_image_path)

    # Process tile
    try:
        batch_start_time = time.time()
        output = segmentor.predict(
            imgs=[tile_image_path],
            save_dir=args.output_dir,
            mode='tile',
            on_gpu=args.gpu,
            crash_on_exception=False
        )
        batch_end_time = time.time()
        batch_time = batch_end_time - batch_start_time
        logger.info(f"Time taken for tile {idx+1}: {batch_time:.2f} seconds")

        # Log processed tile
        log_processed_tile(idx)
        processed_tiles.add(idx)
        processed_count += 1

        # Remove temporary tile image
        os.remove(tile_image_path)

        # Estimate progress
        progress = (processed_count / total_tiles) * 100
        elapsed_time = time.time() - start_time
        estimated_total_time = (elapsed_time / processed_count) * total_tiles
        remaining_time = estimated_total_time - elapsed_time
        formatted_remaining_time = time.strftime("%Hh %Mm %Ss", time.gmtime(remaining_time))
        logger.info(f"Progress: {progress:.2f}% | Estimated remaining time: {formatted_remaining_time}")

    except Exception as e:
        logger.error(f"Error processing tile {idx+1}: {e}")
        # Handle exception as needed
        continue

# End processing
end_time = time.time()
total_time = end_time - start_time
formatted_total_time = time.strftime("%Hh %Mm %Ss", time.gmtime(total_time))
logger.info(f"Total processing time: {formatted_total_time}")
