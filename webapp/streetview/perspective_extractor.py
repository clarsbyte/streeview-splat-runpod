"""Extract pinhole perspective views from equirectangular panorama images."""

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class ImageInfo:
    filename: str
    pano_id: str
    lat: float
    lng: float
    heading: float
    pitch: float
    fov: float
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


def equirect_to_perspective(
    equirect: np.ndarray,
    heading_deg: float,
    pitch_deg: float,
    fov_deg: float,
    out_width: int,
    out_height: int,
) -> np.ndarray:
    """Reproject a region of an equirectangular image to a pinhole perspective view.

    Args:
        equirect: Input equirectangular image (H, W, 3).
        heading_deg: Horizontal look direction in degrees (0=north, 90=east).
        pitch_deg: Vertical look direction in degrees (0=horizon, positive=up).
        fov_deg: Horizontal field of view in degrees.
        out_width: Output image width in pixels.
        out_height: Output image height in pixels.

    Returns:
        Perspective image of shape (out_height, out_width, 3).
    """
    eq_h, eq_w = equirect.shape[:2]

    # Focal length from FOV
    fov_rad = math.radians(fov_deg)
    fx = (out_width / 2.0) / math.tan(fov_rad / 2.0)
    fy = fx  # square pixels

    cx, cy = out_width / 2.0, out_height / 2.0

    # Rotation matrix: heading around Y, pitch around X
    heading_rad = math.radians(heading_deg)
    pitch_rad = math.radians(pitch_deg)

    # Rotation around Y axis (heading / yaw)
    Ry = np.array([
        [math.cos(heading_rad), 0, math.sin(heading_rad)],
        [0, 1, 0],
        [-math.sin(heading_rad), 0, math.cos(heading_rad)],
    ])

    # Rotation around X axis (pitch)
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(pitch_rad), -math.sin(pitch_rad)],
        [0, math.sin(pitch_rad), math.cos(pitch_rad)],
    ])

    R = Ry @ Rx

    # Build pixel grid for the output image
    u = np.arange(out_width, dtype=np.float64)
    v = np.arange(out_height, dtype=np.float64)
    u, v = np.meshgrid(u, v)

    # Ray directions in camera space
    x = (u - cx) / fx
    y = (v - cy) / fy
    z = np.ones_like(x)

    # Stack and normalize
    dirs = np.stack([x, y, z], axis=-1)  # (H, W, 3)
    dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)

    # Rotate to world space
    dirs_world = dirs @ R.T  # (H, W, 3)

    # Convert to spherical coordinates
    dx = dirs_world[..., 0]
    dy = dirs_world[..., 1]
    dz = dirs_world[..., 2]

    # Longitude and latitude
    lon = np.arctan2(dx, dz)  # [-pi, pi]
    lat = np.arcsin(np.clip(dy, -1, 1))  # [-pi/2, pi/2]

    # Map to equirectangular pixel coordinates
    eq_x = ((lon / math.pi + 1.0) / 2.0 * eq_w).astype(np.float32)
    eq_y = ((0.5 - lat / math.pi) * eq_h).astype(np.float32)

    # Wrap x coordinates
    eq_x = eq_x % eq_w

    # Sample from equirectangular image
    perspective = cv2.remap(
        equirect,
        eq_x,
        eq_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    )

    return perspective


def extract_perspectives(
    pano_dir: Path,
    output_dir: Path,
    num_views: int = 8,
    fov_deg: float = 90.0,
    out_size: tuple[int, int] = (1024, 768),
) -> list[ImageInfo]:
    """Extract perspective views from all panoramas in a directory.

    Args:
        pano_dir: Directory containing equirectangular pano images and metadata.json.
        output_dir: Output directory for perspective crops.
        num_views: Number of perspective views per panorama.
        fov_deg: Horizontal field of view in degrees.
        out_size: (width, height) of output perspective images.

    Returns:
        List of ImageInfo for each extracted perspective.
    """
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = pano_dir / "metadata.json"
    with open(metadata_path) as f:
        pano_metadata = json.load(f)

    out_w, out_h = out_size
    fov_rad = math.radians(fov_deg)
    fx = fy = (out_w / 2.0) / math.tan(fov_rad / 2.0)
    cx, cy = out_w / 2.0, out_h / 2.0

    image_infos: list[ImageInfo] = []
    heading_step = 360.0 / num_views

    for pano in pano_metadata:
        pano_path = Path(pano["image_path"])
        if not pano_path.exists():
            print(f"Warning: pano image not found: {pano_path}")
            continue

        equirect = cv2.imread(str(pano_path))
        if equirect is None:
            print(f"Warning: failed to read {pano_path}")
            continue

        for i in range(num_views):
            heading = i * heading_step
            filename = f"{pano['id']}_h{int(heading):03d}.jpg"
            out_path = images_dir / filename

            perspective = equirect_to_perspective(
                equirect, heading, pitch_deg=0.0, fov_deg=fov_deg,
                out_width=out_w, out_height=out_h,
            )
            cv2.imwrite(str(out_path), perspective, [cv2.IMWRITE_JPEG_QUALITY, 95])

            info = ImageInfo(
                filename=filename,
                pano_id=pano["id"],
                lat=pano["lat"],
                lng=pano["lng"],
                heading=heading,
                pitch=0.0,
                fov=fov_deg,
                fx=fx, fy=fy, cx=cx, cy=cy,
                width=out_w, height=out_h,
            )
            image_infos.append(info)

        print(f"Extracted {num_views} views from pano {pano['id']}")

    # Write image metadata
    meta_out = output_dir / "image_metadata.json"
    meta_out.write_text(json.dumps([asdict(i) for i in image_infos], indent=2))
    print(f"Extracted {len(image_infos)} perspective views total")

    return image_infos


def main():
    parser = argparse.ArgumentParser(description="Extract perspective views from panoramas")
    parser.add_argument("--pano_dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--num_views", type=int, default=8)
    parser.add_argument("--fov", type=float, default=90.0)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=768)
    args = parser.parse_args()

    extract_perspectives(
        Path(args.pano_dir), Path(args.output),
        num_views=args.num_views, fov_deg=args.fov,
        out_size=(args.width, args.height),
    )


if __name__ == "__main__":
    main()
