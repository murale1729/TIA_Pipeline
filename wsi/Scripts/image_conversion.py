import tifffile
from PIL import Image
import joblib
import os
import numpy as np

def load_metadata(metadata_path):
    """Load metadata from a pickle file."""
    if os.path.exists(metadata_path):
        with open(metadata_path, 'rb') as f:
            metadata = joblib.load(f)
            print(f"Loaded metadata from {metadata_path}")
            return metadata
    else:
        raise FileNotFoundError(f"Metadata file {metadata_path} not found.")

def convert_png_to_tiff(input_path, output_path, metadata=None, mpp=None):
    """
    Convert a PNG image to TIFF and optionally embed metadata.
    :param input_path: Path to the input PNG image
    :param output_path: Path to save the output TIFF image
    :param metadata: Metadata dictionary from which MPP can be extracted
    :param mpp: Tuple specifying Microns Per Pixel (MPP) for X and Y axes
    :return: Path to the saved TIFF image
    """
    if input_path.endswith(".png"):
        print("Converting PNG to TIFF...")
        img = Image.open(input_path)
        img = img.convert("RGB")  # Ensure it's in the correct mode (RGB)
        
        # Convert to NumPy array
        image_data = np.array(img)
        
        # Extract MPP from metadata or provided MPP
        if metadata and 'mpp' in metadata:
            mpp = metadata['mpp']
        
        # Prepare the TIFF output path
        tiff_output_path = output_path.replace(".png", ".tiff")
        
        # Prepare the metadata description
        description = None
        if mpp:
            description = f'mpp_x={mpp[0]}, mpp_y={mpp[1]}'
            print(f"Embedding MPP into TIFF: {mpp}")
        
        # Save the image as TIFF with embedded metadata
        tifffile.imwrite(tiff_output_path, image_data, description=description)

        print(f"Converted image saved at: {tiff_output_path}")
        return tiff_output_path
    else:
        return input_path
