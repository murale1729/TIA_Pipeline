import os
import sys
import joblib
import logging
import json
import numpy as np
from scipy.spatial import distance
import cv2
from tiatoolbox.utils.misc import imread

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def load_segmentation_results(output_dir):
    """
    Load the segmentation results from the output directory.
    """
    inst_map_path = os.path.join(output_dir, '0.dat')

    if not os.path.exists(inst_map_path):
        logger.error(f"File '0.dat' not found in {output_dir}")
        sys.exit(1)

    logger.info(f"Loading segmentation results from {inst_map_path}")
    try:
        nuclei_predictions = joblib.load(inst_map_path)
    except Exception as e:
        logger.exception(f"Failed to load segmentation results from {inst_map_path}: {e}")
        sys.exit(1)

    logger.info(f"Number of detected nuclei: {len(nuclei_predictions)}")
    return nuclei_predictions

def calculate_metrics(nuclei_predictions, input_img_shape, mpp_value):
    """
    Calculate metrics based on the segmentation results.
    """
    total_area = 0
    total_aspect_ratio = 0
    total_nuclei = len(nuclei_predictions)
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
    """
    Check if two boxes overlap.
    """
    return not (
        box1[2] <= box2[0] or  # box1 right <= box2 left
        box1[0] >= box2[2] or  # box1 left >= box2 right
        box1[3] <= box2[1] or  # box1 bottom <= box2 top
        box1[1] >= box2[3]     # box1 top >= box2 bottom
    )

def overlay_nuclei(input_img, nuclei_predictions, output_dir):
    """
    Visualize the nuclei segmentation by overlaying contours on the original image.
    """
    img_copy = input_img.copy()
    contours = []
    
    # Extract contours from segmentation data
    for _, nucleus in nuclei_predictions.items():
        contour = np.array(nucleus['contour'])
        contours.append(contour)
    
    # Draw contours on the image
    cv2.drawContours(img_copy, contours, -1, (0, 255, 0), 2)

    # Save the image with contours
    output_img_path = os.path.join(output_dir, 'segmentation_overlay.png')
    cv2.imwrite(output_img_path, img_copy)
    logger.info(f"Segmentation overlay image saved at: {output_img_path}")

    return output_img_path

def generate_report_with_overlay(output_dir, input_img_path, mpp_value, output_json):
    """
    Generate a metrics report, overlay image, and save them.
    """
    # Load the input image
    logger.info(f"Loading input image: {input_img_path}")
    input_img = imread(input_img_path)
    if input_img is None:
        logger.error(f"Failed to load input image: {input_img_path}")
        sys.exit(1)

    # Load the segmentation results
    nuclei_predictions = load_segmentation_results(output_dir)

    # Generate and save the overlay image
    logger.info("Generating overlay image...")
    overlay_image_path = overlay_nuclei(input_img, nuclei_predictions, output_dir)

    # Calculate the metrics
    logger.info("Calculating metrics...")
    metrics = calculate_metrics(nuclei_predictions, input_img.shape, mpp_value)

    # Save the metrics to a JSON file
    logger.info(f"Saving metrics report to {output_json}")
    with open(output_json, 'w') as f:
        json.dump(metrics, f, indent=4)

    logger.info(f"Metrics report saved to {output_json}")

if __name__ == "__main__":
    # Parsing input arguments
    import argparse
    parser = argparse.ArgumentParser(description="Generate metrics and overlay image from segmentation results")
    parser.add_argument('--output_dir', type=str, help='Directory with segmentation results', required=True)
    parser.add_argument('--input_img', type=str, help='Path to the original input image', required=True)
    parser.add_argument('--mpp', type=float, default=0.5, help='Microns per pixel (default: 0.5)')
    parser.add_argument('--output_json', type=str, help='Path to save the metrics report JSON', required=True)
    args = parser.parse_args()

    # Generate the report and overlay image
    generate_report_with_overlay(args.output_dir, args.input_img, args.mpp, args.output_json)
