import asyncio
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .pipeline import (
    broadcast,
    create_job,
    create_streetview_job,
    get_result_path,
    get_video_path,
    jobs,
    run_pipeline,
    run_streetview_pipeline,
    ws_connections,
)

app = FastAPI(title="Video/StreetView to 3D Gaussian Splat")

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.post("/api/upload")
async def upload_video(video: UploadFile = File(...), fps: int = Form(default=5)):
    if fps < 1 or fps > 60:
        return JSONResponse({"error": "FPS must be between 1 and 60"}, status_code=400)

    job_id = create_job(fps)
    video_path = get_video_path(job_id)

    with open(video_path, "wb") as f:
        while chunk := await video.read(1024 * 1024):
            f.write(chunk)

    # Start pipeline in background
    asyncio.create_task(run_pipeline(job_id))

    return {"job_id": job_id}


@app.post("/api/streetview")
async def streetview_job(request: Request):
    body = await request.json()
    lat = body.get("lat")
    lng = body.get("lng")
    max_panos = body.get("max_panos", 50)
    num_views = body.get("num_views", 8)

    if lat is None or lng is None:
        return JSONResponse({"error": "lat and lng are required"}, status_code=400)

    job_id = create_streetview_job(lat, lng, max_panos, num_views)
    asyncio.create_task(run_streetview_pipeline(job_id))

    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}/status")
async def job_status(job_id: str):
    if job_id not in jobs:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    job = jobs[job_id]
    return {k: v for k, v in job.items() if k != "job_dir"}


@app.get("/api/jobs/{job_id}/result")
async def job_result(job_id: str):
    if job_id not in jobs:
        return JSONResponse({"error": "Job not found"}, status_code=404)

    result_path = get_result_path(job_id)
    if not result_path:
        return JSONResponse({"error": "Result not ready"}, status_code=404)

    return FileResponse(
        str(result_path),
        media_type="application/octet-stream",
        filename="point_cloud.ply",
    )


@app.websocket("/ws/jobs/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    await websocket.accept()

    if job_id not in ws_connections:
        ws_connections[job_id] = set()
    ws_connections[job_id].add(websocket)

    # Send current status immediately
    if job_id in jobs:
        job = jobs[job_id]
        await websocket.send_json({k: v for k, v in job.items() if k != "job_dir"})

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_connections[job_id].discard(websocket)
