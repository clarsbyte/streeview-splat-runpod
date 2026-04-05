import asyncio
import os
import re
import time
import uuid
from pathlib import Path
from typing import Callable, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# In-memory job store
jobs: dict[str, dict] = {}

# WebSocket connections per job
ws_connections: dict[str, set] = {}


def create_job(fps: int) -> str:
    job_id = uuid.uuid4().hex[:12]
    job_dir = PROJECT_ROOT / "jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "sfm_output" / "images").mkdir(parents=True, exist_ok=True)
    (job_dir / "gsplat_output").mkdir(parents=True, exist_ok=True)

    jobs[job_id] = {
        "status": "queued",
        "stage": "",
        "progress": 0.0,
        "message": "",
        "fps": fps,
        "job_dir": str(job_dir),
    }
    return job_id


def get_video_path(job_id: str) -> Path:
    return PROJECT_ROOT / "jobs" / job_id / "input.mp4"


def get_result_path(job_id: str) -> Optional[Path]:
    job_dir = PROJECT_ROOT / "jobs" / job_id
    # Check multiple possible locations
    for iterations in ["iteration_80000", "iteration_30000", "iteration_10000", "iteration_7000", "iteration_3000", "iteration_2000"]:
        p = job_dir / "gsplat_output" / "point_cloud" / iterations / "point_cloud.ply"
        if p.exists():
            return p
    return None


async def broadcast(job_id: str, data: dict):
    jobs[job_id].update(data)
    if job_id in ws_connections:
        msg = {k: v for k, v in jobs[job_id].items() if k != "job_dir"}
        dead = set()
        for ws in ws_connections[job_id]:
            try:
                await ws.send_json(msg)
            except Exception:
                dead.add(ws)
        ws_connections[job_id] -= dead


async def _read_stream(stream, job_id: str, parse_fn):
    """Read a subprocess stream line-by-line and parse for progress."""
    while True:
        line = await stream.readline()
        if not line:
            break
        text = line.decode("utf-8", errors="replace").strip()
        if text:
            update = parse_fn(text)
            if update:
                await broadcast(job_id, update)


async def run_pipeline(job_id: str):
    job = jobs[job_id]
    job_dir = Path(job["job_dir"])
    fps = job["fps"]
    video_path = job_dir / "input.mp4"
    sfm_dir = job_dir / "sfm_output"
    gsplat_dir = job_dir / "gsplat_output"

    try:
        start_time = time.time()

        def elapsed_str():
            e = int(time.time() - start_time)
            return f"{e // 60}m {e % 60}s"

        def eta_str(progress):
            if progress <= 0:
                return ""
            elapsed = time.time() - start_time
            total_est = elapsed / progress
            remaining = int(total_est - elapsed)
            if remaining < 60:
                return f"~{remaining}s remaining"
            return f"~{remaining // 60}m {remaining % 60}s remaining"

        # Stage 1: Extract frames
        await broadcast(job_id, {"status": "running", "stage": "extracting_frames", "progress": 0.0, "message": "Extracting frames from video..."})

        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-i", str(video_path), "-vf", f"fps={fps},scale='min(1024,iw)':-2",
            "-q:v", "2",
            str(sfm_dir / "images" / "frame_%04d.jpg"),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            await broadcast(job_id, {"status": "error", "message": f"FFmpeg failed: {stderr.decode()[-500:]}"})
            return

        # Deduplicate similar frames to speed up COLMAP
        frames = sorted((sfm_dir / "images").glob("frame_*.jpg"))
        if len(frames) > 60:
            # Keep at most 60 evenly spaced frames
            keep = set(range(0, len(frames), max(1, len(frames) // 60)))
            keep.add(len(frames) - 1)  # always keep last
            for i, f in enumerate(frames):
                if i not in keep:
                    f.unlink()
            frames = sorted((sfm_dir / "images").glob("frame_*.jpg"))
        await broadcast(job_id, {"progress": 0.1, "message": f"Using {len(frames)} frames [{elapsed_str()}]"})

        # Stage 2: MASt3R pose estimation (GPU — much faster than COLMAP)
        await broadcast(job_id, {"stage": "running_mast3r", "progress": 0.15, "message": "Running MASt3R 3D reconstruction (GPU)..."})

        # Create image metadata for video frames (no GPS, just sequential)
        image_dir = sfm_dir / "images"
        frame_files = sorted(image_dir.glob("frame_*.jpg"))

        # Write minimal metadata for mast3r_runner (no GPS for video)
        import json as _json
        video_metadata = []
        for i, f in enumerate(frame_files):
            video_metadata.append({
                "filename": f.name,
                "pano_id": "video",
                "lat": 0.0,
                "lng": 0.0,
                "heading": 0.0,
                "width": 1024,
                "height": 768,
                "fx": 512.0, "fy": 512.0,
                "cx": 512.0, "cy": 384.0,
            })
        meta_path = sfm_dir / "image_metadata.json"
        meta_path.write_text(_json.dumps(video_metadata))

        mast3r_out = job_dir / "mast3r_output"
        mast3r_out.mkdir(parents=True, exist_ok=True)

        proc = await asyncio.create_subprocess_exec(
            "python3", "-m", "webapp.streetview.mast3r_runner",
            "--image_dir", str(image_dir),
            "--metadata", str(meta_path),
            "--output", str(mast3r_out),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(PROJECT_ROOT),
            env={**os.environ, "PYTHONPATH": "/app/mast3r:/app/mast3r/dust3r:" + os.environ.get("PYTHONPATH", "")},
        )

        async def parse_mast3r():
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").strip()
                if "Loading MASt3R" in text:
                    await broadcast(job_id, {"progress": 0.17, "message": f"Loading MASt3R model... [{elapsed_str()}]"})
                elif "Loading" in text and "images" in text:
                    await broadcast(job_id, {"progress": 0.20, "message": f"Loading images... [{elapsed_str()}]"})
                elif "Running inference" in text:
                    await broadcast(job_id, {"progress": 0.25, "message": f"Running pairwise inference (GPU)... [{elapsed_str()}]"})
                elif "Running global alignment" in text:
                    await broadcast(job_id, {"progress": 0.30, "message": f"Optimizing 3D alignment... [{elapsed_str()}]"})
                elif "MASt3R complete" in text:
                    await broadcast(job_id, {"progress": 0.35, "message": f"MASt3R reconstruction done [{elapsed_str()}]"})

        await parse_mast3r()
        await proc.wait()
        if proc.returncode != 0:
            stderr = await proc.stderr.read()
            await broadcast(job_id, {"status": "error", "message": f"MASt3R failed: {stderr.decode()[-500:]}"})
            return

        # Stage 2b: Convert MASt3R output to COLMAP format
        await broadcast(job_id, {"stage": "converting", "progress": 0.35, "message": f"Converting to training format... [{elapsed_str()}]"})

        undistorted_dir = sfm_dir / "undistorted"
        proc = await asyncio.create_subprocess_exec(
            "python3", "-m", "webapp.streetview.mast3r_to_colmap",
            "--mast3r_output", str(mast3r_out),
            "--image_dir", str(image_dir),
            "--output", str(undistorted_dir),
            "--image_metadata", str(meta_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(PROJECT_ROOT),
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            await broadcast(job_id, {"status": "error", "message": f"COLMAP conversion failed: {stderr.decode()[-500:]}"})
            return

        await broadcast(job_id, {"progress": 0.4, "message": f"Reconstruction complete [{elapsed_str()}]"})

        # Stage 3: Gaussian Splat Training
        await broadcast(job_id, {"stage": "training", "progress": 0.4, "message": "Starting Gaussian Splat training..."})

        proc = await asyncio.create_subprocess_exec(
            str(PROJECT_ROOT / "train_speedy_splat.sh"),
            str(undistorted_dir), str(gsplat_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(PROJECT_ROOT),
        )

        await _parse_training_progress(proc, job_id, start_time, progress_base=0.4, progress_range=0.55)
        await proc.wait()
        if proc.returncode != 0:
            stdout = await proc.stdout.read()
            await broadcast(job_id, {"status": "error", "message": f"Training failed: {stdout.decode()[-500:]}"})
            return

        # Done
        result_path = get_result_path(job_id)
        if result_path:
            await broadcast(job_id, {"status": "done", "stage": "complete", "progress": 1.0, "message": f"Training complete! Total time: {elapsed_str()}"})
        else:
            await broadcast(job_id, {"status": "error", "message": "Training finished but no output PLY found."})

    except Exception as e:
        await broadcast(job_id, {"status": "error", "message": str(e)})


# ---------------------------------------------------------------------------
# Street View Pipeline
# ---------------------------------------------------------------------------

def create_streetview_job(lat: float, lng: float, max_panos: int = 50, num_views: int = 8) -> str:
    job_id = uuid.uuid4().hex[:12]
    job_dir = PROJECT_ROOT / "jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    for sub in ["panos", "perspectives", "mast3r_output", "undistorted", "gsplat_output"]:
        (job_dir / sub).mkdir(parents=True, exist_ok=True)

    jobs[job_id] = {
        "status": "queued",
        "stage": "",
        "progress": 0.0,
        "message": "",
        "pipeline": "streetview",
        "lat": lat,
        "lng": lng,
        "max_panos": max_panos,
        "num_views": num_views,
        "job_dir": str(job_dir),
    }
    return job_id


async def _parse_training_progress(proc, job_id: str, start_time: float, progress_base: float, progress_range: float):
    """Shared training progress parser for both video and streetview pipelines."""
    train_start_time = time.time()
    buf = b""
    while True:
        chunk = await proc.stderr.read(4096)
        if not chunk:
            break
        buf += chunk
        parts = re.split(rb"[\r\n]+", buf)
        buf = parts[-1]
        for part in parts[:-1]:
            text = part.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            match = re.search(r"(\d+)/(\d+).*Loss[=:](\S+)", text)
            if not match:
                continue
            current = int(match.group(1))
            total = int(match.group(2))
            loss = match.group(3).rstrip("]")
            train_progress = current / total
            overall = progress_base + train_progress * progress_range
            train_elapsed = time.time() - train_start_time
            if train_progress > 0:
                train_remaining = int(train_elapsed / train_progress - train_elapsed)
                mins, secs = divmod(train_remaining, 60)
                eta = f"~{mins}m {secs}s remaining"
            else:
                eta = ""
            te_mins, te_secs = divmod(int(train_elapsed), 60)
            total_elapsed = int(time.time() - start_time)
            await broadcast(job_id, {
                "progress": overall,
                "message": f"Training: {current}/{total} iterations, Loss: {loss} [{te_mins}m {te_secs}s] {eta}"
            })


async def run_streetview_pipeline(job_id: str):
    job = jobs[job_id]
    job_dir = Path(job["job_dir"])
    lat = job["lat"]
    lng = job["lng"]
    max_panos = job["max_panos"]
    num_views = job["num_views"]
    gsplat_dir = job_dir / "gsplat_output"

    try:
        start_time = time.time()

        def elapsed_str():
            e = int(time.time() - start_time)
            return f"{e // 60}m {e % 60}s"

        # Stage 1: Crawl panoramas (0.0 - 0.15)
        await broadcast(job_id, {
            "status": "running", "stage": "crawling_panos",
            "progress": 0.0, "message": "Finding Street View panoramas..."
        })

        proc = await asyncio.create_subprocess_exec(
            "python3", "-m", "webapp.streetview.pano_crawler",
            "--lat", str(lat), "--lng", str(lng),
            "--output", str(job_dir / "panos"),
            "--max_panos", str(max_panos),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(PROJECT_ROOT),
        )

        async def parse_crawl():
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").strip()
                match = re.search(r"Downloaded (\d+)/(\d+)", text)
                if match:
                    current = int(match.group(1))
                    total = int(match.group(2))
                    progress = 0.0 + (current / total) * 0.15
                    await broadcast(job_id, {
                        "progress": progress,
                        "message": f"Downloading panorama {current}/{total}... [{elapsed_str()}]"
                    })

        await parse_crawl()
        await proc.wait()
        if proc.returncode != 0:
            stderr = await proc.stderr.read()
            await broadcast(job_id, {"status": "error", "message": f"Pano crawl failed: {stderr.decode()[-500:]}"})
            return

        await broadcast(job_id, {"progress": 0.15, "message": f"Panoramas downloaded [{elapsed_str()}]"})

        # Stage 2: Extract perspective views (0.15 - 0.25)
        await broadcast(job_id, {
            "stage": "extracting_perspectives",
            "progress": 0.15, "message": "Extracting perspective views..."
        })

        proc = await asyncio.create_subprocess_exec(
            "python3", "-m", "webapp.streetview.perspective_extractor",
            "--pano_dir", str(job_dir / "panos"),
            "--output", str(job_dir / "perspectives"),
            "--num_views", str(num_views),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(PROJECT_ROOT),
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            await broadcast(job_id, {"status": "error", "message": f"Perspective extraction failed: {stderr.decode()[-500:]}"})
            return

        await broadcast(job_id, {"progress": 0.25, "message": f"Perspective views extracted [{elapsed_str()}]"})

        # Stage 3: MASt3R pose estimation (0.25 - 0.45)
        await broadcast(job_id, {
            "stage": "running_mast3r",
            "progress": 0.25, "message": "Running MASt3R 3D reconstruction..."
        })

        proc = await asyncio.create_subprocess_exec(
            "python3", "-m", "webapp.streetview.mast3r_runner",
            "--image_dir", str(job_dir / "perspectives" / "images"),
            "--metadata", str(job_dir / "perspectives" / "image_metadata.json"),
            "--output", str(job_dir / "mast3r_output"),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(PROJECT_ROOT),
        )

        async def parse_mast3r():
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").strip()
                if "Loading MASt3R" in text:
                    await broadcast(job_id, {"progress": 0.27, "message": f"Loading MASt3R model... [{elapsed_str()}]"})
                elif "Loading" in text and "images" in text:
                    await broadcast(job_id, {"progress": 0.30, "message": f"Loading images... [{elapsed_str()}]"})
                elif "Running inference" in text:
                    await broadcast(job_id, {"progress": 0.33, "message": f"Running pairwise inference... [{elapsed_str()}]"})
                elif "Running global alignment" in text:
                    await broadcast(job_id, {"progress": 0.38, "message": f"Optimizing 3D alignment... [{elapsed_str()}]"})
                elif "MASt3R complete" in text:
                    await broadcast(job_id, {"progress": 0.44, "message": f"MASt3R reconstruction done [{elapsed_str()}]"})

        await parse_mast3r()
        await proc.wait()
        if proc.returncode != 0:
            stderr = await proc.stderr.read()
            await broadcast(job_id, {"status": "error", "message": f"MASt3R failed: {stderr.decode()[-500:]}"})
            return

        await broadcast(job_id, {"progress": 0.45, "message": f"Pose estimation complete [{elapsed_str()}]"})

        # Stage 4: Convert to COLMAP format (0.45 - 0.50)
        await broadcast(job_id, {
            "stage": "converting",
            "progress": 0.45, "message": "Converting to training format..."
        })

        proc = await asyncio.create_subprocess_exec(
            "python3", "-m", "webapp.streetview.mast3r_to_colmap",
            "--mast3r_output", str(job_dir / "mast3r_output"),
            "--image_dir", str(job_dir / "perspectives" / "images"),
            "--output", str(job_dir / "undistorted"),
            "--image_metadata", str(job_dir / "perspectives" / "image_metadata.json"),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(PROJECT_ROOT),
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            await broadcast(job_id, {"status": "error", "message": f"COLMAP conversion failed: {stderr.decode()[-500:]}"})
            return

        await broadcast(job_id, {"progress": 0.50, "message": f"Format conversion complete [{elapsed_str()}]"})

        # Stage 5: Train Gaussian Splat (0.50 - 1.0)
        await broadcast(job_id, {
            "stage": "training",
            "progress": 0.50, "message": "Starting Gaussian Splat training..."
        })

        proc = await asyncio.create_subprocess_exec(
            str(PROJECT_ROOT / "train_speedy_splat.sh"),
            str(job_dir / "undistorted"), str(gsplat_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(PROJECT_ROOT),
        )

        await _parse_training_progress(proc, job_id, start_time, progress_base=0.50, progress_range=0.45)
        await proc.wait()
        if proc.returncode != 0:
            stdout = await proc.stdout.read()
            await broadcast(job_id, {"status": "error", "message": f"Training failed: {stdout.decode()[-500:]}"})
            return

        # Done
        result_path = get_result_path(job_id)
        if result_path:
            await broadcast(job_id, {
                "status": "done", "stage": "complete",
                "progress": 1.0, "message": f"Training complete! Total time: {elapsed_str()}"
            })
        else:
            await broadcast(job_id, {"status": "error", "message": "Training finished but no output PLY found."})

    except Exception as e:
        await broadcast(job_id, {"status": "error", "message": str(e)})
