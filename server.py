# =============================================================================
# server.py — FastAPI WebSocket server
#
# Starts the TD3 training loop in a background daemon thread and streams
# live metrics to connected browser clients over WebSocket.
#
# Run:  python server.py
# Open: http://localhost:8000  (after running: cd frontend && npm run build)
#       or point your browser at http://localhost:5173 with `npm run dev`
# =============================================================================

import asyncio
import json
import queue
import threading

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

from config import WS_HOST, WS_PORT

app = FastAPI()

# Thread-safe queue: training thread puts packets, asyncio drains them
_metrics_queue: queue.Queue = queue.Queue(maxsize=500)


# ── Training thread ────────────────────────────────────────────────────────────
def _training_thread():
    from main import run_training

    def broadcast(packet: dict):
        try:
            _metrics_queue.put_nowait(packet)
        except queue.Full:
            pass  # drop — never block the training thread

    run_training(broadcast_fn=broadcast)


# ── WebSocket connection manager ───────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self._active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self._active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self._active:
            self._active.remove(ws)

    async def broadcast(self, data: str):
        dead = []
        for ws in self._active:
            try:
                await ws.send_text(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._active.remove(ws)


manager = ConnectionManager()


# ── Background coroutine: drain queue → broadcast to all WS clients ────────────
async def _drain_queue():
    while True:
        drained = 0
        while drained < 10:   # drain up to 10 packets per tick
            try:
                packet = _metrics_queue.get_nowait()
                await manager.broadcast(json.dumps(packet))
                drained += 1
            except queue.Empty:
                break
        await asyncio.sleep(0.05)   # 50 ms poll cycle


# ── Startup: launch training thread + drain coroutine ─────────────────────────
@app.on_event("startup")
async def startup():
    t = threading.Thread(target=_training_thread, daemon=True)
    t.start()
    asyncio.create_task(_drain_queue())


# ── WebSocket endpoint ─────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            # Keep the connection alive; client may send "ping"
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)


# ── Serve React production build ───────────────────────────────────────────────
import pathlib as _pathlib

_dist = _pathlib.Path(__file__).parent / "frontend" / "dist"
if _dist.exists():
    app.mount("/assets", StaticFiles(directory=str(_dist / "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        return FileResponse(str(_dist / "index.html"))


if __name__ == "__main__":
    uvicorn.run(app, host=WS_HOST, port=WS_PORT, log_level="info")
