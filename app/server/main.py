import hashlib, logging
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from .. import config as C
from ..cv.engine import engine, ITEMS
from ..agent.graph import ask

log = logging.getLogger("uvicorn.error")
app = FastAPI(title="PPE Q&A")
STATUS, PROG = {}, {}
IMGS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
OK = IMGS | {".mp4", ".mov", ".avi", ".mkv", ".webm"}

def prescan(mid):
    try:
        m = engine.get(mid)
        if m.is_image:
            # Tag image for standard items
            for it in ("hardhat", "vest"):
                engine.tag(m, it)
            engine.annotate(m, 0, "all", "detections")
            PROG[mid] = 1
            STATUS[mid] = "ready"
            return

        for j, it in enumerate(("hardhat", "vest")):
            engine.tag(m, it, cb=lambda f, j=j: PROG.__setitem__(mid, (j + f) / 2))

        # Pre-render evidence frames for detected violations (start, mid, end)
        for it in ("hardhat", "vest"):
            ev = engine.ev.get((m.id, it), {}).get("events", [])
            viols = [e for e in ev if e["state"] == "violation"]
            for k, r in enumerate(viols[:3]):
                engine.annotate(m, r["start_s"], it, "violation start")
                engine.annotate(m, (r["start_s"] + r["end_s"]) / 2, it, "violation mid")
                engine.annotate(m, max(r["start_s"], r["end_s"] - 1 / m.fps), it, "violation end")

        PROG[mid] = 1
        STATUS[mid] = "ready"
    except Exception as e:
        log.exception("prescan failed")
        STATUS[mid] = f"error: {e}"

def start(mid, bg):
    STATUS[mid] = "processing"
    PROG[mid] = 0
    bg.add_task(prescan, mid)

@app.post("/upload")
async def upload(bg: BackgroundTasks, file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in OK:
        raise HTTPException(400, "Unsupported file type. Use a video (mp4, mov, avi, mkv, webm) or image (jpg, png, bmp, webp).")
    data = await file.read()
    mid = hashlib.sha1(data).hexdigest()[:12]
    p = C.UP / f"{mid}{ext}"
    if not p.exists():
        p.write_bytes(data)
    if not STATUS.get(mid, "").startswith(("ready", "processing")):
        start(mid, bg)
    return dict(media_id=mid, is_image=ext in IMGS)

@app.get("/status/{mid}")
def status(mid: str, bg: BackgroundTasks):
    if mid not in STATUS and any(C.UP.glob(mid + ".*")):
        start(mid, bg)   # e.g. after a server restart
    s = STATUS.get(mid, "unknown")
    return dict(status=s.split(":")[0], message=s, progress=round(PROG.get(mid, 0), 2))

@app.get("/timeline/{mid}")
def timeline(mid: str):
    m = engine.get(mid)
    evidence = []
    if m.is_image:
        n = engine.annotate(m, 0, "all", "detections")
        evidence.append(dict(
            url="/evidence/" + n,
            t=0.0,
            caption="Image Analysis & Annotations",
            item="all",
            tag="detection"
        ))
    else:
        seen = set()
        for it in ("hardhat", "vest"):
            ev = engine.ev.get((m.id, it), {}).get("events", [])
            viols = [e for e in ev if e["state"] == "violation"]
            for k, r in enumerate(viols[:3]):
                pts = [
                    ("start", r["start_s"]),
                    ("mid", (r["start_s"] + r["end_s"]) / 2),
                    ("end", max(r["start_s"], r["end_s"] - 1 / m.fps))
                ]
                for tag, sec in pts:
                    sec_rnd = round(sec, 2)
                    if (it, sec_rnd) in seen: continue
                    seen.add((it, sec_rnd))
                    n = engine.annotate(m, sec, it, f"violation {tag}")
                    evidence.append(dict(
                        url="/evidence/" + n,
                        t=sec_rnd,
                        caption=f"{it} violation ({tag}) at {sec:.1f}s",
                        item=it,
                        tag=tag
                    ))
        if not evidence:
            sec = min(1.0, m.dur / 2)
            n = engine.annotate(m, sec, "all", "sample")
            evidence.append(dict(
                url="/evidence/" + n,
                t=round(sec, 2),
                caption=f"Sample frame at {sec:.1f}s",
                item="sample",
                tag="sample"
            ))

    return dict(
        duration=m.dur,
        is_image=m.is_image,
        items={k[1]: v["events"] for k, v in engine.ev.items() if k[0] == mid},
        evidence=evidence
    )

@app.get("/frame/{mid}")
def frame(mid: str, t: float, item: str):
    m = engine.get(mid)
    clean_item = item.lower() if item else "all"
    n = engine.annotate(m, min(max(t, 0), m.dur), clean_item, "view")
    return dict(url="/evidence/" + n, caption=f"{item} at {t:.1f} s")

class Q(BaseModel):
    media_id: str
    question: str

@app.post("/ask")
def ask_route(q: Q):
    if STATUS.get(q.media_id) != "ready":
        raise HTTPException(409, "Still processing")
    try:
        return ask(q.media_id, q.question)
    except Exception as e:
        log.exception("ask failed")
        return dict(
            answer=f"I couldn't analyse that ({type(e).__name__}). Please try again or rephrase.",
            verdict="cannot_determine",
            item="", citations=[], evidence=[], sources=[], coverage={}
        )

app.mount("/evidence", StaticFiles(directory=C.EVID))
app.mount("/", StaticFiles(directory=C.ROOT / "frontend", html=True))
