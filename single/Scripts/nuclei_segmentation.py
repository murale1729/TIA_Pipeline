import argparse
import os
import sys
import joblib
import matplotlib.pyplot as plt
from tiatoolbox.models.engine.nucleus_instance_segmentor import NucleusInstanceSegmentor
from tiatoolbox.utils.misc import download_data, imread
from tiatoolbox.utils.visualization import overlay_prediction_contours
from tiatoolbox.wsicore.wsireader import WSIReader
from skimage import measure
import logging
import torch
import json
import numpy as np
from scipy.spatial import distance
import cv2
import glob
import datetime

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Parsing input arguments
parser = argparse.ArgumentParser(description="Nuclei Segmentation using HoVerNet")
parser.add_argument('--input', type=str, help='Path to normalized image or WSI', required=True)
parser.add_argument('--mask', type=str, help='Path to tissue mask image (binary mask)', required=True)
parser.add_argument('--output_dir', type=str, help='Directory to save output results', required=True)
parser.add_argument('--metadata', type=str, help='Path to metadata.pkl file', required=False)
parser.add_argument('--mode', type=str, default="tile", choices=["wsi", "tile"], help='Processing mode: "wsi" or "tile"')
parser.add_argument('--gpu', action='store_true', help='Use GPU for processing')
parser.add_argument('--default_mpp', type=float, help="Default MPP if not found in metadata", default=0.5)
parser.add_argument('--pretrained_model', type=str, default="hovernet_fast-pannuke", help='Pretrained model to use')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size for processing')
args = parser.parse_args()

# Ensure GPU availability
if args.gpu and not torch.cuda.is_available():
    logger.warning("GPU not available, switching to CPU.")
    args.gpu = False
logger.info(f"Using GPU for processing: {args.gpu}")

logger.debug(f"Input arguments: {args}")

# Validate input paths
if not os.path.exists(args.input):
    logger.error(f"Input file does not exist: {args.input}")
    sys.exit(1)
if not os.path.exists(args.mask):
    logger.error(f"Mask file does not exist: {args.mask}")
    sys.exit(1)
if args.metadata and not os.path.exists(args.metadata):
    logger.error(f"Metadata file does not exist: {args.metadata}")
    sys.exit(1)
# Ensure output directory exists
os.makedirs(args.output_dir, exist_ok=True)

# Load metadata
if args.metadata:
    logger.info(f"Loading metadata from {args.metadata}")
    metadata = joblib.load(args.metadata)
else:
    metadata = {}
    logger.warning("No metadata provided, using default MPP.")

# Ensure MPP is valid and calculate a single MPP value
if 'mpp' not in metadata or not isinstance(metadata['mpp'], (int, float, tuple)):
    logger.warning("Invalid or missing MPP in metadata, using default MPP.")
    mpp_value = args.default_mpp
else:
    mpp = metadata['mpp']
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
    pretrained_model=args.pretrained_model,
    num_loader_workers=2,
    num_postproc_workers=2,
    batch_size=args.batch_size,
    auto_generate_mask=False
)

# Process depending on the mode (wsi or tile)
if args.mode == "wsi":
    logger.info(f"Running segmentation on WSI: {args.input}")
    try:
        wsi = WSIReader.open(args.input)
        output = segmentor.predict(
            imgs=[wsi],
            save_dir=args.output_dir,
            mode='wsi',
            on_gpu=args.gpu,
            crash_on_exception=False
        )
    except Exception as e:
        logger.exception(f"Segmentation failed for WSI: {e}")
        exit(1)
else:
    logger.info(f"Running segmentation on Tile: {args.input}")

    # Load the tissue mask (binary mask)
    logger.info(f"Loading tissue mask from: {args.mask}")
    tissue_mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    if tissue_mask is None:
        logger.error(f"Failed to load tissue mask: {args.mask}")
        sys.exit(1)

    # Ensure the mask is binary (if it's not, threshold it)
    _, tissue_mask_binary = cv2.threshold(tissue_mask, 127, 255, cv2.THRESH_BINARY)

    # Load the input image
    logger.info(f"Loading input image: {args.input}")
    input_img = imread(args.input)
    if input_img is None:
        logger.error(f"Failed to load input image: {args.input}")
        sys.exit(1)

    # Ensure the mask and input image have the same dimensions
    if tissue_mask_binary.shape[:2] != input_img.shape[:2]:
        logger.info(f"Resizing tissue mask to match the input image dimensions: {input_img.shape[:2]}")
        tissue_mask_binary = cv2.resize(
            tissue_mask_binary,
            (input_img.shape[1], input_img.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    # Convert input image to uint8 if it's in a different format
    if input_img.dtype != np.uint8:
        input_img = input_img.astype(np.uint8)

    # Apply the mask to the input image (mask out non-tissue regions)
    logger.info("Applying tissue mask to the input image.")
    masked_img = cv2.bitwise_and(input_img, input_img, mask=tissue_mask_binary)

    # Save masked_img to a temporary file (this ensures we pass a file path to segmentor.predict)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_output_dir = os.path.join(args.output_dir, f"nuclei_run_{timestamp}")
    os.makedirs(unique_output_dir, exist_ok=True)  # Ensure the directory exists

    masked_img_path = os.path.join(unique_output_dir, 'masked_image.png')
    cv2.imwrite(masked_img_path, masked_img)
    logger.info(f"Masked image saved at: {masked_img_path}")

    logger.info(f"Saving Nuclei results in {unique_output_dir}")
    try:
        output = segmentor.predict(
            imgs=[masked_img_path],  # Pass the file path, not the ndarray
            save_dir=unique_output_dir,
            mode='tile',
            on_gpu=args.gpu,
            crash_on_exception=False
        )
    except Exception as e:
        logger.exception(f"Segmentation failed for Tile: {e}")
        exit(1)

logger.debug(f"Segmentation output: {output}")

# Get the correct path to the output file
output_dir_for_image = unique_output_dir
logger.info(f"Segmentation results saved in: {output_dir_for_image}")

# Define the path to the instance map file
# Find .dat files in the output directory
dat_files = glob.glob(os.path.join(output_dir_for_image, '*.dat'))
if not dat_files:
    logger.error(f"No .dat files found in {output_dir_for_image}")
    sys.exit(1)
# Assuming there's only one .dat file per image
inst_map_path = dat_files[0]

# Load the segmentation results
logger.info(f"Loading segmentation results from {inst_map_path}")
try:
    nuclei_predictions = np.load(inst_map_path, allow_pickle=True).item()
except Exception as e:
    logger.exception(f"Failed to load segmentation results: {e}")
    sys.exit(1)

logger.info(f"Number of detected nuclei: {len(nuclei_predictions)}")

# Calculate metrics
def calculate_metrics(nuclei_predictions, input_img_shape, mpp_value):
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
    for nucleus_id, nucleus in nuclei_predictions.items():
        box_area = (nucleus['box'][2] - nucleus['box'][0]) * (nucleus['box'][3] - nucleus['box'][1])
        total_area += box_area

        # Calculate aspect ratio
        width = nucleus['box'][2] - nucleus['box'][0]
        height = nucleus['box'][3] - nucleus['box'][1]
        aspect_ratio = width / height if height != 0 else 0
        total_aspect_ratio += aspect_ratio

        # Confidence score
        confidences.append(nucleus.get('prob', 0))

        # Check for overlaps
        for other_id, other_nucleus in nuclei_predictions.items():
            if nucleus_id == other_id:
                continue  # Skip comparison with itself
            if boxes_overlap(nucleus['box'], other_nucleus['box']):
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
    avg_probability = np.mean(confidences) if confidences else 0
    avg_nearest_neighbor_distance = np.mean(nearest_neighbor_distances) if nearest_neighbor_distances else 0

    # Calculate tile area in mm²
    pixel_area = (mpp_value / 1000) ** 2  # Convert microns to millimeters
    tile_height, tile_width = input_img_shape[:2]
    tile_area = tile_height * tile_width * pixel_area
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

def boxes_overlap(box1, box2):
    return not (
        box1[2] <= box2[0] or  # box1 right <= box2 left
        box1[0] >= box2[2] or  # box1 left >= box2 right
        box1[3] <= box2[1] or  # box1 bottom <= box2 top
        box1[1] >= box2[3]     # box1 top >= box2 bottom
    )

# Calculate the metrics and save to a JSON file
metrics = calculate_metrics(nuclei_predictions, input_img.shape, mpp_value)
metrics_output_path = os.path.join(output_dir_for_image, 'segmentation_metrics.json')
with open(metrics_output_path, 'w') as f:
    json.dump(metrics, f, indent=4)

logger.info(f"Segmentation metrics saved to {metrics_output_path}")
