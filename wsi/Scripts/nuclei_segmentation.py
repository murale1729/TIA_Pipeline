import argparse
import os
import joblib
import matplotlib.pyplot as plt
from tiatoolbox.models.engine.nucleus_instance_segmentor import NucleusInstanceSegmentor
from tiatoolbox.utils.visualization import overlay_prediction_contours
from tiatoolbox.wsicore.wsireader import WSIReader
from skimage import measure
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
parser.add_argument('--metadata', type=str, help='Path to metadata.pkl file', required=False)
parser.add_argument('--mode', type=str, default="tile", choices=["wsi", "tile"], help='Processing mode: "wsi" or "tile"')
parser.add_argument('--gpu', action='store_true', help='Use GPU for processing')
parser.add_argument('--default_mpp', type=float, help="Default MPP if not found in metadata", default=0.5)
args = parser.parse_args()

if not args.gpu:
    args.gpu = torch.cuda.is_available()
logger.info(f"Using GPU for processing: {args.gpu}")

logger.debug(f"Input arguments: {args}")

# Load metadata if provided
if args.metadata:
    logger.info(f"Loading metadata from {args.metadata}")
    metadata = joblib.load(args.metadata)
else:
    metadata = {}
    logger.warning("No metadata provided, using default MPP.")

mpp = metadata.get('mpp', args.default_mpp)

# Ensure MPP is valid and calculate a single MPP value
if isinstance(mpp, tuple) and len(mpp) == 2:
    mpp_value = sum(mpp) / len(mpp)
elif isinstance(mpp, (int, float)):
    mpp_value = float(mpp)
else:
    mpp_value = args.default_mpp
    logger.warning(f"Invalid MPP in metadata, using default MPP: {mpp_value}")

logger.info(f"Microns per pixel (MPP) used: {mpp_value}")

# Initialize NucleusInstanceSegmentor
logger.info("Initializing NucleusInstanceSegmentor")
segmentor = NucleusInstanceSegmentor(
    pretrained_model="hovernet_fast-pannuke",
    num_loader_workers=2,
    num_postproc_workers=2,
    batch_size=4,
    auto_generate_mask=False
)

# Process depending on the mode (wsi or tile)
if args.mode == "wsi":
    logger.info(f"Running segmentation on WSI: {args.input}")
    try:
        # Load WSI with WSIReader
        wsi_reader = WSIReader.open(args.input)
        
        if wsi_reader is None:
            raise ValueError(f"Failed to load WSI: {args.input}")
        
        # Segment the WSI at a specific resolution (use mpp_value)
        resolution = {"mpp": (mpp_value, mpp_value)}
        logger.info(f"Using resolution: {resolution} for WSI segmentation.")
        
        # Perform segmentation on the WSI
        output = segmentor.predict(
            imgs=[wsi_reader],
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

# Define the path to the instance map file
inst_map_path = os.path.join(output_dir_for_image, '0.dat')

# Load the segmentation results
logger.info(f"Loading segmentation results from {inst_map_path}")
nuclei_predictions = joblib.load(inst_map_path)

logger.info(f"Number of detected nuclei: {len(nuclei_predictions)}")

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
        centroids.append(np.array(nucleus['centroid']))

    # Calculate metrics for each nucleus
    for _, nucleus in nuclei_predictions.items():
        box_area = (nucleus['box'][2] - nucleus['box'][0]) * (nucleus['box'][3] - nucleus['box'][1])
        total_area += box_area

        # Calculate aspect ratio
        width = nucleus['box'][2] - nucleus['box'][0]
        height = nucleus['box'][3] - nucleus['box'][1]
        aspect_ratio = width / height
        total_aspect_ratio += aspect_ratio

        # Confidence score
        confidences.append(nucleus.get('prob', 0))

        # Check for overlaps (bounding boxes overlap)
        for _, other_nucleus in nuclei_predictions.items():
            if np.array_equal(nucleus['box'], other_nucleus['box']):
                continue  # Skip comparison with itself
            if np.any(np.intersect1d(nucleus['box'], other_nucleus['box'])):
                nuclei_with_overlaps += 1
                break  # Count overlap only once per nucleus

        # Nearest neighbor distance
        distances = distance.cdist([nucleus['centroid']], centroids, 'euclidean')
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
    avg_probability = np.mean(confidences)
    avg_nearest_neighbor_distance = np.mean(nearest_neighbor_distances)

    # Assuming a given area of the tile (in mm²), calculate density
    tile_area = 1  # In mm², adjust according to your actual data
    nuclei_density = total_nuclei / tile_area

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
