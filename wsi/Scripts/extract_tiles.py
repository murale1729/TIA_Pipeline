from tiatoolbox.tools.patchextraction import PatchExtractor
from tiatoolbox.wsicore.wsireader import WSIReader
import argparse

parser = argparse.ArgumentParser(description="Tile Extraction")
parser.add_argument('--input', type=str, help='Path to WSI file')
#parser.add_argument('--heatmap', type=str, help='Path to heatmap image')
parser.add_argument('--output', type=str, help='Output directory to save extracted tiles')

args = parser.parse_args()

# Load the WSI
wsi = WSIReader.open(args.input)

# Extract high-density tiles based on a fixed grid
patch_extractor = PatchExtractor(tile_shape=(512, 512), stride_shape=(256, 256))
tiles = patch_extractor.extract(wsi)

# Save the tiles
for i, tile in enumerate(tiles):
    tile.save(f'{args.output}/tile_{i}.png')
