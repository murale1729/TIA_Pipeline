import argparse
import os
import joblib
import logging
import torch
import json
import numpy as np
from scipy.spatial import distance
from tiatoolbox.models.engine.nucleus_instance_segmentor import NucleusInstanceSegmentor
from tiatoolbox.models.engine.semantic_segmentor import IOSegmentorConfig
from tiatoolbox.wsicore.wsireader import WSIReader

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Parsing input arguments
parser = argparse.ArgumentParser(description="Nuclei Segmentation using HoVerNet with optimized GPU utilization")
parser.add_argument('--input', type=str, help='Path to normalized image or WSI', required=True)
parser.add_argument('--output_dir', type=str, help='Directory to save output results', required=True)
parser.add_argument('--mode', type=str, default="wsi", choices=["wsi", "tile"], help='Processing mode: "wsi" or "tile"')
parser.add_argument('--gpu', action='store_true', help='Use GPU for processing')
parser.add_argument('--default_mpp', type=float, help="Default MPP if not found in metadata", default=0.5)
args = parser.parse_args()

# Check for GPU usage
if not args.gpu:
    args.gpu = torch.cuda.is_available()

device = torch.device("cuda" if args.gpu else "cpu")
logger.info(f"Using device: {device}")

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
    num_loader_workers=8,  # Increased number of data loader workers
    num_postproc_workers=8,  # Increased number of post-processing workers
    batch_size=16,  # Increased batch size
    auto_generate_mask=False
)

# Ensure that the model is moved to the GPU
segmentor.model.to(device)

# Process depending on the mode (wsi or tile)
if args.mode == "wsi":
    logger.info(f"Running segmentation on WSI: {args.input}")
    try:
        # Define the patch shapes (adjusted to larger sizes)
        patch_input_shape = [540, 540]   # Larger input patch size
        patch_output_shape = [160, 160]  # Corresponding output patch size

        # Create an IOConfig object for WSI processing
        ioconfig = IOSegmentorConfig(
            input_resolutions=[{"mpp": mpp_value}],
            output_resolutions=[{"mpp": mpp_value}],
            save_resolution={"mpp": mpp_value},
            patch_input_shape=patch_input_shape,
            patch_output_shape=patch_output_shape
        )

        # Run the segmentation
        output = segmentor.predict(
            imgs=[args.input],
            save_dir=args.output_dir,
            mode='wsi',
            ioconfig=ioconfig,
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
image_basename = os.path.splitext(os.path.basename(args.input))[0]
output_dir_for_image = os.path.join(args.output_dir, image_basename)
logger.info(f"Segmentation results saved in: {output_dir_for_image}")

# Check if the output directory exists
if not os.path.exists(output_dir_for_image):
    logger.error(f"Output directory not found: {output_dir_for_image}")
    exit(1)

# Find the instance map file (e.g., 'inst_map.dat')
inst_map_path = os.path.join(output_dir_for_image, 'inst_map.dat')
if not os.path.exists(inst_map_path):
    logger.error(f"No instance map file found in {output_dir_for_image}")
    exit(1)

# Load the segmentation results
try:
    logger.info(f"Loading segmentation results from {inst_map_path}")
    inst_map = joblib.load(inst_map_path)
    logger.info(f"Instance map loaded with shape: {inst_map.shape}")
except Exception as e:
    logger.error(f"Failed to load segmentation results: {e}")
    exit(1)

# Calculate metrics
def calculate_metrics(inst_map):
    from skimage.measure import regionprops

    props = regionprops(inst_map)
    total_area = 0
    total_aspect_ratio = 0
    total_nuclei = len(props)
    nearest_neighbor_distances = []

    type_distribution = {
        'neoplastic_epithelial': 0,
        'inflammatory': 0,
        'connective': 0,
        'dead_cells': 0,
        'other': 0
    }

    centroids = [prop.centroid for prop in props]

    for prop in props:
        area = prop.area
        total_area += area

        # Calculate aspect ratio
        minr, minc, maxr, maxc = prop.bbox
        width = maxc - minc
        height = maxr - minr
        aspect_ratio = width / height if height != 0 else 0
        total_aspect_ratio += aspect_ratio

        # Nearest neighbor distance
        distances = distance.cdist([prop.centroid], centroids, 'euclidean')
        if len(distances.flatten()) > 1:
            nearest_distance = np.partition(distances.flatten(), 1)[1]  # Skip distance to itself
            nearest_neighbor_distances.append(nearest_distance)

        # Nucleus type classification (if available)
        nucleus_type = prop.label  # Adjust based on your data
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
        'average_nearest_neighbor_distance': avg_nearest_neighbor_distance,
        'nuclei_density': nuclei_density
    }

    return metrics

# Calculate the metrics and save to a JSON file
metrics = calculate_metrics(inst_map)
metrics_output_path = os.path.join(output_dir_for_image, 'segmentation_metrics.json')
with open(metrics_output_path, 'w') as f:
    json.dump(metrics, f, indent=4)

logger.info(f"Segmentation metrics saved to {metrics_output_path}")
