"""Run MASt3R for pose estimation and dense point cloud generation from perspective images."""

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class MASt3RResult:
    poses: list[np.ndarray]       # List of 4x4 camera-to-world matrices
    intrinsics: list[np.ndarray]  # List of 3x3 intrinsic matrices
    points3d: np.ndarray          # (N, 3) dense point cloud
    colors: np.ndarray            # (N, 3) RGB colors [0-255]
    image_names: list[str]        # Ordered image filenames


def _gps_to_enu(lat: float, lng: float, origin_lat: float, origin_lng: float) -> tuple[float, float, float]:
    """Convert GPS coordinates to local East-North-Up (ENU) meters centered on origin."""
    R = 6371000.0  # Earth radius in meters
    dlat = math.radians(lat - origin_lat)
    dlng = math.radians(lng - origin_lng)
    north = dlat * R
    east = dlng * R * math.cos(math.radians(origin_lat))
    return east, north, 0.0


def _heading_to_rotation(heading_deg: float) -> np.ndarray:
    """Convert a compass heading to a 3x3 rotation matrix (rotation around Y axis)."""
    rad = math.radians(heading_deg)
    return np.array([
        [math.cos(rad), 0, math.sin(rad)],
        [0, 1, 0],
        [-math.sin(rad), 0, math.cos(rad)],
    ])


def _build_spatial_pairs(image_metadata: list[dict]) -> list[tuple[int, int]]:
    """Build minimal image pairs to keep MASt3R fast and within memory.

    Strategy:
    - Same-pano: only pair neighboring views (adjacent headings)
    - Cross-pano: only pair views with the SAME heading from adjacent panos
    - Cap total pairs to avoid OOM
    """
    n = len(image_metadata)

    pano_groups: dict[str, list[int]] = {}
    for i, meta in enumerate(image_metadata):
        pano_groups.setdefault(meta["pano_id"], []).append(i)

    pairs = set()

    # Intra-pano: only adjacent views (e.g. h000-h090, h090-h180, etc.)
    for indices in pano_groups.values():
        for i in range(len(indices)):
            j = (i + 1) % len(indices)
            a, b = min(indices[i], indices[j]), max(indices[i], indices[j])
            pairs.add((a, b))

    # Cross-pano: same heading only, between consecutive panos
    pano_ids = list(pano_groups.keys())
    for k in range(len(pano_ids) - 1):
        pid_a, pid_b = pano_ids[k], pano_ids[k + 1]
        headings_a = {image_metadata[i]["heading"]: i for i in pano_groups[pid_a]}
        headings_b = {image_metadata[i]["heading"]: i for i in pano_groups[pid_b]}
        for h in headings_a:
            if h in headings_b:
                a, b = min(headings_a[h], headings_b[h]), max(headings_a[h], headings_b[h])
                pairs.add((a, b))

    # Hard cap
    pairs_list = list(pairs)
    if len(pairs_list) > 80:
        import random
        random.shuffle(pairs_list)
        pairs_list = pairs_list[:80]

    print(f"Built {len(pairs_list)} image pairs from {n} images")
    return pairs_list


def run_mast3r(
    image_dir: Path,
    metadata_path: Path,
    output_dir: Path,
    device: str = "cuda",
) -> MASt3RResult:
    """Run MASt3R reconstruction on a set of perspective images.

    Args:
        image_dir: Directory containing perspective images.
        metadata_path: Path to image_metadata.json from perspective_extractor.
        output_dir: Directory to save results.
        device: Torch device ("cuda" or "cpu").

    Returns:
        MASt3RResult with poses, intrinsics, and dense point cloud.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(metadata_path) as f:
        image_metadata = json.load(f)

    image_paths = [str(image_dir / m["filename"]) for m in image_metadata]

    # Import MASt3R (assumes mast3r is on PYTHONPATH)
    from mast3r.model import AsymmetricMASt3R
    from dust3r.inference import inference
    from dust3r.utils.image import load_images
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

    print("Loading MASt3R model...")
    model = AsymmetricMASt3R.from_pretrained(
        "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    ).to(device)

    print(f"Loading {len(image_paths)} images...")
    images = load_images(image_paths, size=512)

    # Build pairs using sliding window — every image paired with its neighbors
    # This guarantees every image index appears in at least one pair
    n = len(images)
    winsize = min(3, n - 1)
    pair_indices = set()
    for i in range(n):
        for j in range(i + 1, min(i + winsize + 1, n)):
            pair_indices.add((i, j))
    # Symmetrize: add both (i,j) and (j,i) as MASt3R is asymmetric
    pairs = []
    for i, j in pair_indices:
        pairs.append((images[i], images[j]))
        pairs.append((images[j], images[i]))

    print(f"Running inference on {len(pairs)} pairs ({n} images, window={winsize})...")
    output = inference(pairs, model, device, batch_size=1)

    print("Running global alignment...")
    scene = global_aligner(
        output, device=device,
        mode=GlobalAlignerMode.ModularPointCloudOptimizer,
    )

    # Set GPS priors as initial pose estimates
    origin_lat = image_metadata[0]["lat"]
    origin_lng = image_metadata[0]["lng"]

    known_poses = []
    for meta in image_metadata:
        east, north, up = _gps_to_enu(meta["lat"], meta["lng"], origin_lat, origin_lng)
        R = _heading_to_rotation(meta["heading"])
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = [east, up, north]  # ENU -> OpenGL convention (Y-up)
        known_poses.append(torch.from_numpy(pose).float().to(device))

    try:
        scene.preset_pose(known_poses)
        scene.compute_global_alignment(init="known_poses", niter=300)
    except Exception as e:
        print(f"Warning: known_poses init failed ({e}), falling back to MST init")
        scene.compute_global_alignment(init="mst", niter=300)

    # Extract results
    poses_tensor = scene.get_im_poses().detach().cpu().numpy()  # (N, 4, 4)
    pts3d_list = scene.get_pts3d()  # list of (H, W, 3) tensors
    intrinsics_tensor = scene.get_intrinsics().detach().cpu().numpy()  # (N, 3, 3)

    # Collect dense point cloud with colors
    all_pts = []
    all_colors = []
    for i, pts in enumerate(pts3d_list):
        pts_np = pts.detach().cpu().numpy().reshape(-1, 3)
        # Load image to get colors
        img = images[i]["img"]
        if torch.is_tensor(img):
            img = img.detach().cpu().numpy()
        if img.ndim == 4:
            img = img[0]
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        # Normalize to 0-255
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        colors = img.reshape(-1, 3)

        # Filter out invalid points (near zero or very far)
        valid = np.isfinite(pts_np).all(axis=1) & (np.linalg.norm(pts_np, axis=1) < 1000)
        all_pts.append(pts_np[valid])
        all_colors.append(colors[valid])

    points3d = np.concatenate(all_pts, axis=0)
    colors = np.concatenate(all_colors, axis=0)

    image_names = [m["filename"] for m in image_metadata]

    result = MASt3RResult(
        poses=[poses_tensor[i] for i in range(len(image_metadata))],
        intrinsics=[intrinsics_tensor[i] for i in range(len(image_metadata))],
        points3d=points3d,
        colors=colors,
        image_names=image_names,
    )

    # Save result to disk
    np.save(str(output_dir / "poses.npy"), poses_tensor)
    np.save(str(output_dir / "intrinsics.npy"), intrinsics_tensor)
    np.save(str(output_dir / "points3d.npy"), points3d)
    np.save(str(output_dir / "colors.npy"), colors)
    (output_dir / "image_names.json").write_text(json.dumps(image_names))

    print(f"MASt3R complete: {len(image_names)} cameras, {len(points3d)} points")
    return result


def main():
    parser = argparse.ArgumentParser(description="Run MASt3R pose estimation")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    run_mast3r(Path(args.image_dir), Path(args.metadata), Path(args.output), args.device)


if __name__ == "__main__":
    main()
