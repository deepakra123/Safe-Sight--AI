import cv2
from pathlib import Path
from .. import config as C
from ultralytics import YOLO
from .openvocab import OpenVocab

LOCK = C.LOCK
IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ITEMS = {"hardhat": ("Hardhat", "NO-Hardhat"), "vest": ("Safety Vest", "NO-Safety Vest"), "mask": ("Mask", "NO-Mask")}
BAND = {"hardhat": (0, .35), "vest": (.2, .75), "mask": (0, .5)}   # where on the person box the evidence must fall
RED, GRN, AMB = (0, 0, 255), (0, 255, 0), (0, 165, 255)
COL = {
    "Person": (255, 191, 0),
    "Hardhat": (0, 160, 0),
    "Safety Vest": (0, 255, 0),
    "Mask": (0, 160, 0),
    "NO-Hardhat": (0, 0, 255),
    "NO-Safety Vest": (0, 0, 255),
    "NO-Mask": (0, 0, 255)
}

class Media:
    def __init__(s, path):
        s.path, s.id = str(path), Path(path).stem
        s.is_image = Path(path).suffix.lower() in IMG
        if s.is_image:
            s.img = cv2.imread(s.path)
            if s.img is None: raise ValueError("Cannot read this image")
            s.h, s.w = s.img.shape[:2]; s.fps, s.n = 1.0, 1
        else:
            s.cap = cv2.VideoCapture(s.path)
            s.fps = s.cap.get(cv2.CAP_PROP_FPS) or 25.0; s.n = int(s.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            s.w, s.h = int(s.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(s.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if not s.cap.isOpened() or s.n < 1: raise ValueError("Cannot read this video")
        s.dur = s.n / s.fps
    def frame(s, i):
        if s.is_image: return s.img.copy()
        i = min(max(i, 0), s.n - 1)
        with LOCK:
            for k in range(5):   # last frames can fail to decode; step back
                s.cap.set(cv2.CAP_PROP_POS_FRAMES, max(i - k, 0)); ok, im = s.cap.read()
                if ok: return im
        raise RuntimeError("cannot read frame")

class Engine:
    def __init__(s): s.model, s.dc, s.media, s.ev, s.ov = None, {}, {}, {}, OpenVocab()
    def get(s, mid):
        if mid not in s.media: s.media[mid] = Media(next(C.UP.glob(mid + ".*")))
        return s.media[mid]
    def dets(s, m, i):
        k = (m.id, i)
        if k not in s.dc:
            with LOCK:
                if s.model is None: s.model = YOLO(str(C.MODEL))
                r = s.model.predict(m.frame(i), conf=C.CONF, verbose=False)[0]
            b = r.boxes
            s.dc[k] = [dict(cls=r.names[int(c)], conf=float(p), box=[float(v) for v in xy])
                       for xy, p, c in zip(b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy())]
        return s.dc[k]
    def people(s, m, i):
        ds = s.dets(m, i)
        ps = [dict(box=p["box"], tiny=p["box"][3] - p["box"][1] < .12 * m.h, cropped=p["box"][1] <= .01 * m.h,
                   ev={k: [] for k in ITEMS}) for p in ds if p["cls"] == "Person" and p["conf"] >= C.PERSON_CONF]
        for d in ds:   # give each PPE box to the tightest person box whose head/torso band contains it
            for it, names in ITEMS.items():
                if d["cls"] not in names: continue
                cx, cy, best = (d["box"][0] + d["box"][2]) / 2, (d["box"][1] + d["box"][3]) / 2, None
                for p in ps:
                    x1, y1, x2, y2 = p["box"]; h = y2 - y1; lo, hi = BAND[it]
                    if x1 <= cx <= x2 and y1 + lo * h <= cy <= y1 + hi * h and (best is None or (x2 - x1) * h < best[0]):
                        best = ((x2 - x1) * h, p)
                if best: best[1]["ev"][it].append(d)
        for p in ps:
            p["st"] = {}
            for it, (pos, neg) in ITEMS.items():
                if not p["ev"][it]: p["st"][it] = ("uncertain", 0.0); continue
                b = max(p["ev"][it], key=lambda d: d["conf"])
                st = "violation" if b["cls"] == neg else "compliant"
                if st == "violation" and ((it == "hardhat" and p["cropped"]) or b["conf"] < C.VIOL_CONF): st = "uncertain"
                p["st"][it] = (st, b["conf"])
        return ps
    def state(s, m, i, item):   # scene-level: violation / compliant / uncertain / no_person
        ps = [p for p in s.people(m, i) if not p["tiny"]]
        if not ps: return "no_person", 0.0
        if item in ITEMS: sts = [p["st"][item] for p in ps]
        else:
            img = m.frame(i); sts = [s.ov.person_state(img, p["box"], item) for p in ps]
        v = [c for t, c in sts if t == "violation"]
        if v: return "violation", max(v)
        if all(t == "compliant" for t, _ in sts): return "compliant", min(c for _, c in sts)
        return "uncertain", 0.0
    def probe(s, m, i, item):   # voted probe over neighbouring frames, used by the binary search
        return sum(s.state(m, min(max(i + o * 2, 0), m.n - 1), item)[0] == "violation" for o in (-1, 0, 1)) >= 2
    def tag(s, m, item, cb=None):
        if (m.id, item) in s.ev: return s.ev[(m.id, item)]
        step = max(1, round(m.fps * C.STRIDE)); idx = list(range(0, m.n, step))
        if idx[-1] != m.n - 1: idx.append(m.n - 1)
        st = []                                             # coarse pass
        for k, i in enumerate(idx):
            st.append(s.state(m, i, item))
            if cb: cb(k / len(idx))
        pred = [t == "violation" for t, _ in st]; iv, start = [], (0 if pred[0] else None)
        for k in range(1, len(idx)):                        # binary-search each state flip
            if pred[k] == pred[k - 1]: continue
            lo, hi = idx[k - 1], idx[k]
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if s.probe(m, mid, item) == pred[k - 1]: lo = mid
                else: hi = mid
            if pred[k]: start = hi
            else: iv.append((start, hi)); start = None
        if start is not None: iv.append((start, m.n))
        def runs(flag):
            out, a = [], None
            for k, i in enumerate(idx):
                if flag(st[k][0]) and a is None: a = i
                if not flag(st[k][0]) and a is not None: out.append((a, i)); a = None
            return out + ([(a, m.n)] if a is not None else [])
        def clean(iv):   # merge gaps < 0.5 s, drop events < 0.5 s
            out = []
            for a, b in sorted(iv):
                if out and (a - out[-1][1]) / m.fps < .5: out[-1][1] = max(out[-1][1], b)
                else: out.append([a, b])
            return [(a, b) for a, b in out if m.is_image or (b - a) / m.fps >= .5]
        def conf(a, b):
            c = [x for (t, x), i in zip(st, idx) if a <= i < b and t == "violation"]
            return round(sum(c) / len(c), 2) if c else 0.0
        ev = [dict(item=item, state="violation", start_s=a / m.fps, end_s=b / m.fps, conf=conf(a, b)) for a, b in clean(iv)]
        ev += [dict(item=item, state="uncertain", start_s=a / m.fps, end_s=b / m.fps, conf=0.0)
               for a, b in clean(runs(lambda t: t == "uncertain"))]
        cov = {t: round(100 * sum(x[0] == t for x in st) / len(st)) for t in ("violation", "compliant", "uncertain", "no_person")}
        s.ev[(m.id, item)] = dict(events=ev, coverage=cov, frames=len(set(idx)))
        return s.ev[(m.id, item)]

    def annotate(s, m, t, item="all", label=""):
        i = min(m.n - 1, max(0, int(round(t * m.fps))))
        img = m.frame(i)
        base = img.copy()
        is_known = item in ITEMS or item in ("all", "overview", "detection", "sample")
        for d in s.dets(m, i):
            c = COL.get(d["cls"])
            if c and (is_known or d["cls"] == "Person"):
                x1, y1, x2, y2 = map(int, d["box"])
                cv2.rectangle(img, (x1, y1), (x2, y2), c, 3)
                cv2.putText(img, f"{d['cls']} {d['conf']:.2f}", (x1, max(y1 - 6, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
        if not is_known:   # open-vocab: colour each person box by the item's state
            for p in s.people(m, i):
                if p["tiny"]: continue
                t2, _ = s.ov.person_state(base, p["box"], item)
                x1, y1, x2, y2 = map(int, p["box"])
                c = {"violation": RED, "compliant": GRN}.get(t2, AMB)
                cv2.rectangle(img, (x1, y1), (x2, y2), c, 3)
                cv2.putText(img, f"{item}: {t2}", (x1, y2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        title = f"{item} {label}".strip() if label else item
        cv2.putText(img, f"{title}  t={i / m.fps:.2f}s  frame={i}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        safe_item = item.replace(" ", "_")
        name = f"{m.id}_{safe_item}_{i}.jpg"
        cv2.imwrite(str(C.EVID / name), img)
        return name

engine = Engine()
