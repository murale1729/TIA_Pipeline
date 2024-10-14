#!/usr/bin/env python

import os
import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from tiatoolbox.models import DeepFeatureExtractor
from tiatoolbox.models.architecture.vanilla import CNNBackbone

# Command-line arguments
parser = argparse.ArgumentParser(description="Deep Feature Extraction for Nuclei Segmentation Results")
parser.add_argument('--input', type=str, help='Path to the nuclei segmentation result file (0.dat)', required=True)
parser.add_argument('--output', type=str, help='Path to save extracted features as CSV', required=True)
parser.add_argument('--gpu', action='store_true', help='Use GPU for processing')

args = parser.parse_args()

# Load nuclei segmentation result from the .dat file
if os.path.exists(args.input):
    nuclei_result = joblib.load(args.input)
    print(f"Loaded segmentation result from {args.input}")
    print(nuclei_result)
else:
    raise FileNotFoundError(f"Nuclei segmentation result file {args.input} not found.")

# Initialize ResNet50 model for feature extraction
model = CNNBackbone("resnet50")
extractor = DeepFeatureExtractor(batch_size=16, model=model, num_loader_workers=4)

# Placeholder for features
features_list = []

# Loop through each instance (nucleus) in the segmentation result
for instance_id, instance_data in nuclei_result.items():
    # Extract the bounding box of the nucleus and the instance mask
    instance_mask = instance_data["mask"]
    
    # Resize the mask to match the input size for ResNet (224x224)
    resized_instance = np.resize(instance_mask, (224, 224, 3))  # Assuming it's a 3-channel mask
    
    # Run the ResNet model to extract features
    extracted_features = extractor.model.predict(resized_instance)
    
    # Flatten the feature vector and add it to the list
    features_list.append(extracted_features.flatten())

# Convert features to a DataFrame
df = pd.DataFrame(features_list)
print(df)

# Ensure the output directory exists
output_path = Path(args.output)
output_dir = output_path.parent
output_dir.mkdir(parents=True, exist_ok=True)

# Save extracted features to CSV
df.to_csv(args.output, index=False)
print(f"Feature extraction completed. Results saved to {args.output}")
