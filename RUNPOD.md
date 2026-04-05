# RunPod Deployment Guide

## 1. Create Pod

- **Template:** `runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04`
- **GPU:** L40S (recommended) or A40
- **Disk:** 50GB+ container volume
- **Expose ports:** 8888 (API — shares Jupyter port)

## 2. Setup (run once)

SSH in or use the Jupyter terminal:

```bash
cd /app  # or wherever you want
git clone https://github.com/clarsbyte/streeview-splat-runpod.git .

# Install dependencies
pip install fastapi uvicorn python-multipart websockets aiofiles streetlevel opencv-python-headless pyproj plyfile "numpy<2"

# Clone and install MASt3R (only needed for Street View pipeline)
git clone --recursive https://github.com/naver/mast3r /app/mast3r
cd /app/mast3r && pip install -r requirements.txt && pip install -r dust3r/requirements.txt
cd /app

# Pre-download MASt3R model weights (~2GB)
PYTHONPATH="/app/mast3r:/app/mast3r/dust3r" python3 -c "from mast3r.model import AsymmetricMASt3R; AsymmetricMASt3R.from_pretrained('naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric')"

# Clone speedy-splat
git clone https://github.com/j-alex-hanern/speedy-splat.git /app/speedy-splat
cd /app/speedy-splat && pip install -r requirements.txt
cd /app

# Install COLMAP (for video pipeline)
apt-get update && apt-get install -y colmap
pip install pycolmap

# Bump file descriptor limit (MASt3R can exhaust defaults)
ulimit -n 65536
```

## 3. Start the server

```bash
cd /app
export PYTHONPATH="/app/mast3r:/app/mast3r/dust3r:$PYTHONPATH"
ulimit -n 65536
python3 -m uvicorn webapp.app:app --host 0.0.0.0 --port 8888
```

The API is now live at `https://<pod-id>-8888.proxy.runpod.net/`.

## 4. API Endpoints

### Video → Splat

```bash
# Upload video
curl -X POST https://<pod-id>-8888.proxy.runpod.net/api/upload \
  -F "video=@walkthrough.mp4" \
  -F "fps=2"
# Returns: {"job_id": "abc123"}

# Poll status
curl https://<pod-id>-8888.proxy.runpod.net/api/jobs/abc123/status

# Download result PLY
curl -o model.ply https://<pod-id>-8888.proxy.runpod.net/api/jobs/abc123/result
```

### Street View → Splat

```bash
# Start job
curl -X POST https://<pod-id>-8888.proxy.runpod.net/api/streetview \
  -H "Content-Type: application/json" \
  -d '{"lat": 37.7749, "lng": -122.4194, "max_panos": 10, "num_views": 6}'
# Returns: {"job_id": "def456"}

# Poll status (same endpoint)
curl https://<pod-id>-8888.proxy.runpod.net/api/jobs/def456/status

# Download result PLY (same endpoint)
curl -o model.ply https://<pod-id>-8888.proxy.runpod.net/api/jobs/def456/result
```

## 5. Update code

```bash
cd /app
git pull origin main
# Restart the server (Ctrl+C then re-run the uvicorn command)
```

## 6. Frontend connection

Set in your straightline `.env.local`:

```
RUNPOD_URL=https://<pod-id>-8888.proxy.runpod.net
```
