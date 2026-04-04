FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3.10-dev \
    ffmpeg \
    colmap \
    git cmake build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
RUN pip3 install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121

RUN pip3 install plyfile tqdm packaging

# Street View pipeline dependencies
RUN pip3 install streetlevel opencv-python-headless pyproj numpy

# Web app dependencies
COPY webapp/requirements.txt /tmp/webapp-requirements.txt
RUN pip3 install -r /tmp/webapp-requirements.txt

# Copy project
WORKDIR /app
COPY . /app/

# Patch torch cpp_extension if needed
RUN sed -i 's/from pkg_resources import packaging/import packaging.version/' \
    $(python3 -c "import torch.utils.cpp_extension as m; print(m.__file__)" 2>/dev/null || echo "/dev/null") 2>/dev/null || true

# Build CUDA extensions
RUN cd /app/speedy-splat && \
    pip3 install --no-build-isolation submodules/diff-gaussian-rasterization && \
    pip3 install --no-build-isolation submodules/simple-knn

# MASt3R + dust3r for Street View pose estimation
RUN git clone --recursive https://github.com/naver/mast3r /app/mast3r && \
    cd /app/mast3r && \
    pip3 install -r requirements.txt && \
    pip3 install -r dust3r/requirements.txt

# Pre-download MASt3R model weights to avoid cold-start delay
RUN python3 -c "import sys; sys.path.insert(0, '/app/mast3r'); sys.path.insert(0, '/app/mast3r/dust3r'); from mast3r.model import AsymmetricMASt3R; AsymmetricMASt3R.from_pretrained('naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric')"

ENV PYTHONPATH="/app/mast3r:/app/mast3r/dust3r:${PYTHONPATH}"

# Create jobs directory
RUN mkdir -p /app/jobs

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "webapp.app:app", "--host", "0.0.0.0", "--port", "8000"]
