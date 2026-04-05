"""GPU-accelerated COLMAP pipeline using pycolmap."""
import shutil
from pathlib import Path

import pycolmap


def run_colmap(image_path: str, output_dir: str, use_gpu: bool = True):
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    database_path = output_dir / "database.db"
    sparse_dir = output_dir / "sparse"
    undistorted_dir = output_dir / "undistorted"

    sparse_dir.mkdir(parents=True, exist_ok=True)
    undistorted_dir.mkdir(parents=True, exist_ok=True)

    device = pycolmap.Device.auto if use_gpu else pycolmap.Device.cpu

    # Step 1: Feature extraction (GPU)
    print("STAGE: feature_extraction", flush=True)
    pycolmap.extract_features(
        database_path=str(database_path),
        image_path=str(image_path),
        device=device,
        camera_mode=pycolmap.CameraMode.SINGLE,
    )

    # Step 2: Feature matching (GPU)
    print("STAGE: feature_matching", flush=True)
    pycolmap.match_exhaustive(
        database_path=str(database_path),
        device=device,
    )

    # Step 3: Sparse reconstruction
    print("STAGE: sparse_reconstruction", flush=True)
    mapper_options = pycolmap.IncrementalPipelineOptions()
    mapper_options.ba_global_max_num_iterations = 10
    mapper_options.ba_local_max_num_iterations = 10
    maps = pycolmap.incremental_mapping(
        database_path=str(database_path),
        image_path=str(image_path),
        output_path=str(sparse_dir),
        options=mapper_options,
    )

    if not maps:
        raise RuntimeError("COLMAP failed to create any sparse model")

    # Step 4: Undistort images
    print("STAGE: undistortion", flush=True)
    sparse_model_path = sparse_dir / "0"
    if not sparse_model_path.exists():
        # Check if data is directly in sparse/
        for f in ["cameras.bin", "images.bin", "points3D.bin"]:
            src = sparse_dir / f
            if src.exists():
                sparse_model_path.mkdir(exist_ok=True)
                shutil.move(str(src), str(sparse_model_path / f))

    pycolmap.undistort_images(
        output_path=str(undistorted_dir),
        input_path=str(sparse_model_path),
        image_path=str(image_path),
    )

    # Fix sparse/0 in undistorted if needed
    undist_sparse = undistorted_dir / "sparse"
    if undist_sparse.exists() and not (undist_sparse / "0").exists():
        (undist_sparse / "0").mkdir(exist_ok=True)
        for f in undist_sparse.glob("*.bin"):
            shutil.move(str(f), str(undist_sparse / "0" / f.name))

    print("STAGE: done", flush=True)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python colmap_runner.py <image_path> <output_dir> [--no-gpu]")
        sys.exit(1)
    use_gpu = "--no-gpu" not in sys.argv
    run_colmap(sys.argv[1], sys.argv[2], use_gpu=use_gpu)
