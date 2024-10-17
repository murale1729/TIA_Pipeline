import argparse
import os
import joblib
import matplotlib.pyplot as plt
from tiatoolbox.models.engine.nucleus_instance_segmentor import NucleusInstanceSegmentor
from tiatoolbox.utils.visualization import overlay_prediction_contours
from tiatoolbox.wsicore.wsireader import WSIReader
import logging
import torch
import json
import numpy as np
from scipy.spatial import distance

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Parsing input arguments
parser = argparse.ArgumentParser(description="Nuclei Segmentation using HoVerNet")
parser.add_argument('--input', type=str, help='Path to normalized image or WSI', required=True)
parser.add_argument('--output_dir', type=str, help='Directory to save output results', required=True)
parser.add_argument('--mode', type=str, default="wsi", choices=["wsi", "tile"], help='Processing mode: "wsi" or "tile"')
parser.add_argument('--gpu', action='store_true', help='Use GPU for processing')
parser.add_argument('--default_mpp', type=float, help="Default MPP if not found in metadata", default=0.5)
args = parser.parse_args()

# GPU availability check
if not args.gpu:
    args.gpu = torch.cuda.is_available()
logger.info(f"Using GPU for processing: {args.gpu}")

logger.debug(f"Input arguments: {args}")

# Load the WSI and extract metadata directly from the SVS file
wsi_reader = WSIReader.open(args.input)
metadata = wsi_reader.info.as_dict()
logger.info(f"WSI Metadata: {metadata}")

# Extract MPP (Microns Per Pixel) from the metadata
mpp = metadata.get('mpp', None)

# Ensure MPP is valid and calculate a single MPP value
if isinstance(mpp, (tuple, list)) and len(mpp) == 2:
    mpp_value = sum(mpp) / len(mpp)
elif isinstance(mpp, (int, float)):
    mpp_value = float(mpp)
else:
    mpp_value = args.default_mpp
    logger.warning(f"MPP not found in metadata or invalid format, using default MPP: {mpp_value}")

logger.info(f"Microns per pixel (MPP) used: {mpp_value}")

# Initialize NucleusInstanceSegmentor with increased batch size and worker counts
logger.info("Initializing NucleusInstanceSegmentor")
segmentor = NucleusInstanceSegmentor(
    pretrained_model="hovernet_fast-pannuke",
    num_loader_workers= 16,   # Increased number of data loader workers
    num_postproc_workers= 16, # Increased number of post-processing workers
    batch_size= 64,          # Increased batch size
    auto_generate_mask=False
)

# Ensure that the model is moved to the GPU if requested
if args.gpu:
    device = torch.device("cuda")
    segmentor.model.to(device)
else:
    device = torch.device("cpu")
    segmentor.model.to(device)

# Process depending on the mode (wsi or tile)
if args.mode == "wsi":
    logger.info(f"Running segmentation on WSI: {args.input}")
    try:
        # Define the resolution using MPP
        resolution = {"mpp": mpp_value}
        logger.info(f"Using resolution: {resolution} for WSI segmentation.")

        # Run the segmentation
        output = segmentor.predict(
            imgs=[args.input],
            save_dir=args.output_dir,
            mode='wsi',
            resolution=resolution,
            on_gpu=args.gpu,
            crash_on_exception=False
        )

    except Exception as e:
        logger.error(f"Segmentation failed for WSI: {e}")
        exit(1)
else:
    # Tile mode
    logger.info(f"Running segmentation on Tile: {args.input}")
    try:
        output = segmentor.predict(
            imgs=[args.input],
            save_dir=args.output_dir,
            mode='tile',
            on_gpu=args.gpu,
            crash_on_exception=False
        )
    except Exception as e:
        logger.error(f"Segmentation failed for Tile: {e}")
        exit(1)

logger.debug(f"Segmentation output: {output}")

# Get the correct path to the output file
output_dir_for_image = args.output_dir
logger.info(f"Segmentation results saved in: {output_dir_for_image}")

# Check if the output directory contains segmentation results
seg_result_files = os.listdir(output_dir_for_image)
if not seg_result_files:
    logger.error(f"No segmentation result files found in {output_dir_for_image}")
    exit(1)

# Find the instance map file (usually named '0.dat' or 'inst_map.dat')
inst_map_path = os.path.join(output_dir_for_image, '0.dat')
if not os.path.exists(inst_map_path):
    # Try alternative filename
    inst_map_path = os.path.join(output_dir_for_image, 'inst_map.dat')
    if not os.path.exists(inst_map_path):
        logger.error(f"Segmentation result file not found: {inst_map_path}")
        exit(1)

# Load the segmentation results
try:
    logger.info(f"Loading segmentation results from {inst_map_path}")
    nuclei_predictions = joblib.load(inst_map_path)
    logger.info(f"Number of detected nuclei: {len(nuclei_predictions)}")
except FileNotFoundError:
    logger.error(f"Segmentation result file not found: {inst_map_path}")
    exit(1)
except Exception as e:
    logger.error(f"Failed to load segmentation results: {e}")
    exit(1)

# Calculate metrics
def calculate_metrics(nuclei_predictions):
    total_area = 0
    total_aspect_ratio = 0
    total_nuclei = len(nuclei_predictions)
    total_probability = 0
    nearest_neighbor_distances = []
    nuclei_with_overlaps = 0

    type_distribution = {
        'neoplastic_epithelial': 0,
        'inflammatory': 0,
        'connective': 0,
        'dead_cells': 0,
        'other': 0
    }

    confidences = []
    centroids = []

    # Gather centroids for nearest neighbor calculation
    for _, nucleus in nuclei_predictions.items():
        centroid = nucleus.get('centroid')
        if centroid is not None:
            centroids.append(np.array(centroid))

    # Calculate metrics for each nucleus
    for _, nucleus in nuclei_predictions.items():
        box = nucleus.get('box')
        centroid = nucleus.get('centroid')
        if box is None or centroid is None:
            continue  # Skip if essential data is missing

        width = box[2] - box[0]
        height = box[3] - box[1]
        if height == 0:
            aspect_ratio = 0
        else:
            aspect_ratio = width / height
        total_aspect_ratio += aspect_ratio

        box_area = width * height
        total_area += box_area

        # Confidence score
        confidences.append(nucleus.get('prob', 0))

        # Check for overlaps
        # Note: Overlaps are complex to compute accurately; this is a simplified version
        # For a more accurate calculation, consider using spatial data structures
        for _, other_nucleus in nuclei_predictions.items():
            if nucleus == other_nucleus:
                continue  # Skip comparison with itself
            other_box = other_nucleus.get('box')
            if other_box is None:
                continue
            # Check for box overlap
            if (box[0] < other_box[2] and box[2] > other_box[0] and
                box[1] < other_box[3] and box[3] > other_box[1]):
                nuclei_with_overlaps += 1
                break  # Count overlap only once per nucleus

        # Nearest neighbor distance
        distances = distance.cdist([centroid], centroids, 'euclidean')
        if len(distances.flatten()) > 1:
            nearest_distance = np.partition(distances.flatten(), 1)[1]  # Skip distance to itself
            nearest_neighbor_distances.append(nearest_distance)

        # Nucleus type classification
        nucleus_type = nucleus.get('type', None)
        if nucleus_type == 1:
            type_distribution['neoplastic_epithelial'] += 1
        elif nucleus_type == 2:
            type_distribution['inflammatory'] += 1
        elif nucleus_type == 3:
            type_distribution['connective'] += 1
        elif nucleus_type == 4:
            type_distribution['dead_cells'] += 1
        else:
            type_distribution['other'] += 1

    avg_area = total_area / total_nuclei if total_nuclei > 0 else 0
    avg_aspect_ratio = total_aspect_ratio / total_nuclei if total_nuclei > 0 else 0
    avg_probability = np.mean(confidences) if confidences else 0
    avg_nearest_neighbor_distance = np.mean(nearest_neighbor_distances) if nearest_neighbor_distances else 0

    # Assuming a given area of the tile (in mm²), calculate density
    # Adjust 'tile_area' according to the actual area covered by the image
    tile_area = 1  # Placeholder value; replace with actual area if available
    nuclei_density = total_nuclei / tile_area if tile_area > 0 else 0

    metrics = {
        'total_nuclei': total_nuclei,
        'nucleus_type_distribution': type_distribution,
        'average_nucleus_area': avg_area,
        'average_aspect_ratio': avg_aspect_ratio,
        'nearest_neighbor_distance': avg_nearest_neighbor_distance,
        'nuclei_density': nuclei_density,
        'confidence_score_distribution': {
            'average_confidence': avg_probability,
            'low_confidence_count': len([c for c in confidences if c < 0.5])
        },
        'nuclei_with_overlaps': nuclei_with_overlaps
    }

    return metrics

# Calculate the metrics and save to a JSON file
metrics = calculate_metrics(nuclei_predictions)
metrics_output_path = os.path.join(output_dir_for_image, 'segmentation_metrics.json')
with open(metrics_output_path, 'w') as f:
    json.dump(metrics, f, indent=4)

logger.info(f"Segmentation metrics saved to {metrics_output_path}")
