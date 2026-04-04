#!/bin/bash
set -e

# Usage: ./streetview_to_gsplat.sh <lat> <lng> <output_dir> [max_panos] [num_views]
#
# Runs the full Street View -> Gaussian Splat pipeline:
#   1. Crawl Street View panoramas
#   2. Extract perspective views
#   3. Run MASt3R pose estimation
#   4. Convert to COLMAP format
#   5. Train Gaussian Splat

LAT="$1"
LNG="$2"
OUTPUT_DIR="$3"
MAX_PANOS="${4:-50}"
NUM_VIEWS="${5:-8}"

if [ -z "$LAT" ] || [ -z "$LNG" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: $0 <lat> <lng> <output_dir> [max_panos] [num_views]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Street View to Gaussian Splat Pipeline ==="
echo "Location: ($LAT, $LNG)"
echo "Output: $OUTPUT_DIR"
echo "Max panos: $MAX_PANOS, Views per pano: $NUM_VIEWS"
echo ""

# Stage 1: Crawl panoramas
echo "=== Stage 1/5: Crawling Street View panoramas ==="
python3 -m webapp.streetview.pano_crawler \
    --lat "$LAT" --lng "$LNG" \
    --output "$OUTPUT_DIR/panos" \
    --max_panos "$MAX_PANOS"

# Stage 2: Extract perspective views
echo ""
echo "=== Stage 2/5: Extracting perspective views ==="
python3 -m webapp.streetview.perspective_extractor \
    --pano_dir "$OUTPUT_DIR/panos" \
    --output "$OUTPUT_DIR/perspectives" \
    --num_views "$NUM_VIEWS"

# Stage 3: MASt3R pose estimation
echo ""
echo "=== Stage 3/5: Running MASt3R pose estimation ==="
python3 -m webapp.streetview.mast3r_runner \
    --image_dir "$OUTPUT_DIR/perspectives/images" \
    --metadata "$OUTPUT_DIR/perspectives/image_metadata.json" \
    --output "$OUTPUT_DIR/mast3r_output"

# Stage 4: Convert to COLMAP format
echo ""
echo "=== Stage 4/5: Converting to COLMAP format ==="
python3 -m webapp.streetview.mast3r_to_colmap \
    --mast3r_output "$OUTPUT_DIR/mast3r_output" \
    --image_dir "$OUTPUT_DIR/perspectives/images" \
    --output "$OUTPUT_DIR/undistorted" \
    --image_metadata "$OUTPUT_DIR/perspectives/image_metadata.json"

# Stage 5: Train Gaussian Splat
echo ""
echo "=== Stage 5/5: Training Gaussian Splat ==="
"$SCRIPT_DIR/train_speedy_splat.sh" "$OUTPUT_DIR/undistorted" "$OUTPUT_DIR/gsplat_output"

echo ""
echo "=== Pipeline complete! ==="
echo "Result: $OUTPUT_DIR/gsplat_output/point_cloud/"
