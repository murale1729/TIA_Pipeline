import os
import sys
import joblib
import logging
import json
import numpy as np
from scipy.spatial import distance
import cv2
import matplotlib.pyplot as plt
from tiatoolbox.wsicore.wsireader import WSIReader
from tiatoolbox.utils.misc import imread
from tiatoolbox.utils.visualization import overlay_prediction_contours

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Patch extraction parameters
bb = 128  # box size for patch extraction around each nucleus

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
    # Define the color dictionary for nuclei types
    color_dict = {
        0: ("background", (255, 165, 0)),  # Orange for background
        1: ("neoplastic epithelial", (255, 0, 0)),  # Red
        2: ("Inflammatory", (255, 255, 0)),  # Yellow
        3: ("Connective", (0, 255, 0)),  # Green
        4: ("Dead", (0, 0, 0)),  # Black
        5: ("non-neoplastic epithelial", (0, 0, 255)),  # Blue
        6: ("Unknown Type", (128, 128, 128)),  # Gray for unknown types
    }

    # Assign default type for unknown types
    for nucleus in nuclei_predictions.values():
        nucleus_type = nucleus.get("type", None)
        if nucleus_type not in color_dict:
            nucleus["type"] = 6  # Assign to 'Unknown Type'

    # Create the overlay image
    overlaid_predictions = overlay_prediction_contours(
        canvas=input_img,
        inst_dict=nuclei_predictions,
        draw_dot=False,
        type_colours=color_dict,
        line_thickness=4,
    )

    # Save the image with contours
    output_img_path = os.path.join(output_dir, 'segmentation_overlay.png')
    cv2.imwrite(output_img_path, overlaid_predictions)
    logger.info(f"Segmentation overlay image saved at: {output_img_path}")

    return output_img_path

def visualize_patches(nuclei_predictions, wsi, rng, output_dir, num_examples=4):
    """
    Visualize and save small patches around randomly selected nuclei with overlaid contours.
    """
    # Define the coloring dictionary
    color_dict = {
        0: ("background", (255, 165, 0)),
        1: ("neoplastic epithelial", (255, 0, 0)),
        2: ("Inflammatory", (255, 255, 0)),
        3: ("Connective", (0, 255, 0)),
        4: ("Dead", (0, 0, 0)),
        5: ("non-neoplastic epithelial", (0, 0, 255)),
        6: ("Unknown Type", (128, 128, 128)),  # Gray for unknown types
    }

    # Create a directory to save patches
    patches_dir = os.path.join(output_dir, 'nuclei_patches')
    os.makedirs(patches_dir, exist_ok=True)

    # Create a list of nucleus IDs to sample from
    nuc_id_list = list(nuclei_predictions.keys())

    fig = plt.figure(figsize=(12, 6))
    
    for i in range(num_examples):  # showing a few examples
        selected_nuc_id = nuc_id_list[rng.integers(0, len(nuclei_predictions))]
        sample_nuc = nuclei_predictions[selected_nuc_id]
        cent = np.int32(sample_nuc["centroid"])  # centroid position in WSI coordinate system
        contour = sample_nuc["contour"]  # nucleus contour points in WSI coordinate system
        contour -= (cent - bb // 2)  # nucleus contour points in the small patch coordinate system

        # Reading the nucleus small window neighborhood
        nuc_patch = wsi.read_rect(cent - bb // 2, bb, resolution=0.25, units="mpp", coord_space="resolution")
        overlaid_patch = cv2.drawContours(nuc_patch.copy(), [contour], -1, (255, 255, 0), 2)

        # Save the patches
        nucleus_type = sample_nuc.get("type", 6)
        type_name = color_dict[nucleus_type][0]
        base_filename = f"nucleus_{selected_nuc_id}_type_{type_name}"
        original_patch_path = os.path.join(patches_dir, f"{base_filename}_original.png")
        overlaid_patch_path = os.path.join(patches_dir, f"{base_filename}_overlay.png")
        cv2.imwrite(original_patch_path, cv2.cvtColor(nuc_patch, cv2.COLOR_RGB2BGR))
        cv2.imwrite(overlaid_patch_path, cv2.cvtColor(overlaid_patch, cv2.COLOR_RGB2BGR))

        logger.info(f"Saved original patch: {original_patch_path}")
        logger.info(f"Saved overlaid patch: {overlaid_patch_path}")

        # Plot the results
        plt.subplot(2, num_examples, i + 1)
        plt.imshow(nuc_patch)
        plt.axis("off")

        plt.subplot(2, num_examples, i + num_examples + 1)
        plt.imshow(overlaid_patch)
        plt.axis("off")
        plt.title(type_name)
    
    plt.tight_layout()
    # Save the figure with patches
    patches_figure_path = os.path.join(output_dir, 'nuclei_patches.png')
    plt.savefig(patches_figure_path)
    logger.info(f"Saved patches figure: {patches_figure_path}")
    plt.show()

def generate_report_with_overlay(output_dir, input_img_path, mpp_value, output_json, wsi_file_name=None):
    """
    Generate a metrics report, overlay image, visualize patches, and save them.
    """
    # Load the input image
    logger.info(f"Loading input image: {input_img_path}")
    input_img = imread(input_img_path)
    if input_img is None:
        logger.error(f"Failed to load input image: {input_img_path}")
        sys.exit(1)

    # Load the segmentation results
    nuclei_predictions = load_segmentation_results(output_dir)

    # Print unique types in nuclei_predictions
    unique_types = set()
    for nucleus in nuclei_predictions.values():
        unique_types.add(nucleus.get("type", None))
    logger.info(f"Unique nucleus types in data: {unique_types}")

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

    # Optionally, display and save patches around selected nuclei
    if wsi_file_name:
        rng = np.random.default_rng()  # random number generator for selecting nuclei
        wsi = WSIReader.open(wsi_file_name)
        visualize_patches(nuclei_predictions, wsi, rng, output_dir)

if __name__ == "__main__":
    # Parsing input arguments
    import argparse
    parser = argparse.ArgumentParser(description="Generate metrics, overlay image, and visualize patches from segmentation results")
    parser.add_argument('--output_dir', type=str, help='Directory with segmentation results', required=True)
    parser.add_argument('--input_img', type=str, help='Path to the original input image', required=True)
    parser.add_argument('--mpp', type=float, default=0.5, help='Microns per pixel (default: 0.5)')
    parser.add_argument('--output_json', type=str, help='Path to save the metrics report JSON', required=True)
    parser.add_argument('--wsi_file_name', type=str, help='Path to the whole slide image (WSI) file', required=False)
    args = parser.parse_args()

    # Generate the report, overlay image, and visualize patches
    generate_report_with_overlay(args.output_dir, args.input_img, args.mpp, args.output_json, args.wsi_file_name)
