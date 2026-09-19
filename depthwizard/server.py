"""FastAPI app: serves the viewer, processed scenes, and the upload → job → scene API."""
import json, os, re, shutil, threading, traceback, uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.gzip import GZipMiddleware

STATIC = Path(__file__).parent / "static"


def create_app(data_dir):
    data_dir = Path(data_dir).resolve()
    scenes_dir, uploads_dir = data_dir / "scenes", data_dir / "uploads"
    scenes_dir.mkdir(parents=True, exist_ok=True); uploads_dir.mkdir(parents=True, exist_ok=True)
    jobs = {}
    run_lock = threading.Lock()  # one job at a time: bounded memory, others wait in the queue
    app = FastAPI(title="DepthWizard")
    app.add_middleware(GZipMiddleware, minimum_size=4096, compresslevel=4)

    @app.get("/api/scenes")
    def list_scenes():
        out = []
        for d in sorted(scenes_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            f = d / "scene.json"
            if f.exists():
                m = json.loads(f.read_text())
                out.append(dict(id=d.name, url=f"./scenes/{d.name}", title=m.get("title") or d.name,
                                height_kind=m.get("height_kind")))
        return out

    @app.post("/api/jobs")
    async def create_job(image: UploadFile = File(...), reference: UploadFile | None = File(None),
                         dem: UploadFile | None = File(None), gcps: UploadFile | None = File(None),
                         gsd: float | None = Form(None)):
        jid = uuid.uuid4().hex[:10]
        jdir = uploads_dir / jid; jdir.mkdir()
        paths = {}
        for key, up in (("image", image), ("reference", reference), ("dem", dem), ("gcps", gcps)):
            if up is None or not up.filename:
                continue
            p = jdir / f"{key}{Path(up.filename).suffix.lower()}"
            with open(p, "wb") as fh:
                shutil.copyfileobj(up.file, fh)
            paths[key] = p
        title = Path(image.filename).stem
        jobs[jid] = dict(id=jid, state="queued", message="Queued", title=title)

        def run():
            from .pipeline import process  # imported lazily: pulls in torch
            job = jobs[jid]
            if run_lock.locked():
                job.update(message="Waiting for the previous job to finish")
            with run_lock:
                _run(job)

        def _run(job):
            from .pipeline import process
            try:
                job.update(state="running")
                out = scenes_dir / f"{re.sub(r'[^A-Za-z0-9._-]+', '_', title)[:60]}-{jid}"
                process(paths["image"], out, reference=paths.get("reference"), dem=paths.get("dem"),
                        gcps=paths.get("gcps"), gsd=gsd, title=title,
                        progress=lambda msg: job.update(message=msg))
                job.update(state="done", message="Done", scene_url=f"./scenes/{out.name}")
            except Exception as e:
                traceback.print_exc()
                job.update(state="error", message=f"{type(e).__name__}: {e}")

        threading.Thread(target=run, daemon=True).start()
        return jobs[jid]

    @app.get("/api/jobs/{jid}")
    def job_status(jid: str):
        if jid not in jobs:
            raise HTTPException(404, "unknown job")
        return jobs[jid]

    app.mount("/scenes", StaticFiles(directory=scenes_dir), name="scenes")
    if STATIC.exists():
        app.mount("/", StaticFiles(directory=STATIC, html=True), name="viewer")
    else:
        @app.get("/")
        def no_viewer():
            return JSONResponse({"error": "viewer not built: run `npm install && npm run build` in web/"}, 500)
    return app
