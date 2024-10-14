from tiatoolbox.wsicore.wsireader import WSIReader
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Read WSI")
parser.add_argument('--input', type=str, help='Path to WSI file')
parser.add_argument('--output', type=str, help='Path to save output thumbnail')

args = parser.parse_args()

# Load the WSI
wsi = WSIReader.open(args.input)

# Generate a thumbnail
thumbnail = wsi.slide_thumbnail(1.0)

# Save the thumbnail
plt.imsave(args.output, thumbnail)
