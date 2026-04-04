"""Convert MASt3R output to COLMAP binary format for speedy-splat consumption."""

import argparse
import json
import shutil
import struct
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def rotmat2qvec(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a COLMAP quaternion (w, x, y, z)."""
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
    ]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def write_cameras_bin(path: Path, width: int, height: int, fx: float, fy: float, cx: float, cy: float):
    """Write cameras.bin with a single PINHOLE camera."""
    with open(path, "wb") as f:
        # num_cameras
        f.write(struct.pack("<Q", 1))
        # camera_id, model_id (PINHOLE=1), width, height
        f.write(struct.pack("<iiQQ", 1, 1, width, height))
        # params: fx, fy, cx, cy
        f.write(struct.pack("<dddd", fx, fy, cx, cy))


def write_images_bin(path: Path, poses: list[np.ndarray], image_names: list[str]):
    """Write images.bin with camera poses.

    Each pose is a 4x4 camera-to-world matrix. COLMAP stores world-to-camera.
    """
    with open(path, "wb") as f:
        # num_images
        f.write(struct.pack("<Q", len(poses)))
        for i, (pose, name) in enumerate(zip(poses, image_names)):
            # Convert camera-to-world (c2w) to world-to-camera (w2c)
            w2c = np.linalg.inv(pose)
            R = w2c[:3, :3]
            t = w2c[:3, 3]
            qvec = rotmat2qvec(R)

            # image_id, qvec (w,x,y,z), tvec (x,y,z), camera_id
            f.write(struct.pack("<i", i + 1))
            f.write(struct.pack("<dddd", *qvec))
            f.write(struct.pack("<ddd", *t))
            f.write(struct.pack("<i", 1))  # camera_id = 1

            # image name (null-terminated)
            f.write(name.encode("utf-8"))
            f.write(b"\x00")

            # num_points2D = 0 (no 2D-3D correspondences needed)
            f.write(struct.pack("<Q", 0))


def write_points3d_bin(path: Path, points: np.ndarray, colors: np.ndarray, max_points: int = 100000):
    """Write points3D.bin with subsampled point cloud."""
    n = len(points)
    if n > max_points:
        indices = np.random.choice(n, max_points, replace=False)
        points = points[indices]
        colors = colors[indices]
        n = max_points

    with open(path, "wb") as f:
        f.write(struct.pack("<Q", n))
        for i in range(n):
            xyz = points[i]
            rgb = np.clip(colors[i], 0, 255).astype(np.uint8)
            # point3D_id, xyz, rgb, error, track_length=0
            f.write(struct.pack("<Q", i + 1))
            f.write(struct.pack("<ddd", *xyz))
            f.write(struct.pack("<BBB", *rgb))
            f.write(struct.pack("<d", 0.0))  # error
            f.write(struct.pack("<Q", 0))    # track_length


def write_points3d_ply(path: Path, points: np.ndarray, colors: np.ndarray, max_points: int = 100000):
    """Write points3D.ply for speedy-splat (avoids runtime bin->ply conversion)."""
    n = len(points)
    if n > max_points:
        indices = np.random.choice(n, max_points, replace=False)
        points = points[indices]
        colors = colors[indices]

    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ]

    normals = np.zeros_like(points, dtype=np.float32)
    rgb = np.clip(colors, 0, 255).astype(np.uint8)

    elements = np.empty(len(points), dtype=dtype)
    for i, (attr_name, _) in enumerate(dtype[:3]):
        elements[attr_name] = points[:, i].astype(np.float32)
    for i, (attr_name, _) in enumerate(dtype[3:6]):
        elements[attr_name] = normals[:, i]
    for i, (attr_name, _) in enumerate(dtype[6:]):
        elements[attr_name] = rgb[:, i]

    vertex_element = PlyElement.describe(elements, "vertex")
    PlyData([vertex_element]).write(str(path))


def convert_to_colmap(
    mast3r_output_dir: Path,
    image_source_dir: Path,
    output_dir: Path,
    image_metadata_path: Path | None = None,
):
    """Convert MASt3R output to COLMAP-format directory structure.

    Args:
        mast3r_output_dir: Directory with poses.npy, intrinsics.npy, points3d.npy, colors.npy
        image_source_dir: Directory with perspective images
        output_dir: Output directory (will contain images/ and sparse/0/)
        image_metadata_path: Optional path to image_metadata.json for camera params
    """
    poses = np.load(str(mast3r_output_dir / "poses.npy"))
    intrinsics = np.load(str(mast3r_output_dir / "intrinsics.npy"))
    points3d = np.load(str(mast3r_output_dir / "points3d.npy"))
    colors = np.load(str(mast3r_output_dir / "colors.npy"))

    with open(mast3r_output_dir / "image_names.json") as f:
        image_names = json.load(f)

    # Get camera dimensions from metadata or intrinsics
    if image_metadata_path and image_metadata_path.exists():
        with open(image_metadata_path) as f:
            meta = json.load(f)
        width = meta[0]["width"]
        height = meta[0]["height"]
        fx = meta[0]["fx"]
        fy = meta[0]["fy"]
        cx = meta[0]["cx"]
        cy = meta[0]["cy"]
    else:
        # Derive from MASt3R intrinsics (first camera)
        K = intrinsics[0]
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        width = int(cx * 2)
        height = int(cy * 2)

    # Create directory structure
    images_out = output_dir / "images"
    sparse_dir = output_dir / "sparse" / "0"
    images_out.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    # Copy images
    for name in image_names:
        src = image_source_dir / name
        dst = images_out / name
        if src.exists():
            shutil.copy2(str(src), str(dst))

    # Write COLMAP binary files
    write_cameras_bin(sparse_dir / "cameras.bin", width, height, fx, fy, cx, cy)
    write_images_bin(sparse_dir / "images.bin", [poses[i] for i in range(len(poses))], image_names)
    write_points3d_bin(sparse_dir / "points3D.bin", points3d, colors)
    write_points3d_ply(sparse_dir / "points3D.ply", points3d, colors)

    print(f"COLMAP conversion complete: {len(image_names)} images, {len(points3d)} points")
    print(f"Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert MASt3R output to COLMAP format")
    parser.add_argument("--mast3r_output", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--image_metadata", type=str, default=None)
    args = parser.parse_args()

    convert_to_colmap(
        Path(args.mast3r_output),
        Path(args.image_dir),
        Path(args.output),
        Path(args.image_metadata) if args.image_metadata else None,
    )


if __name__ == "__main__":
    main()
