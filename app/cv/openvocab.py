import io, os, time, traceback, requests, torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from .. import config as C

NAME = "openai/clip-vit-base-patch32"
MARGIN = 0.02
TTL = 86400
norm = torch.nn.functional.normalize

# Curated queries and text prompts for common open-vocabulary PPE items
PPE_SPECS = {
    "glove": {
        "body_part": "hands",
        "queries_pos": [
            "work gloves",
            "heavy duty safety work gloves",
            "construction work gloves",
            "leather work gloves",
            "protective work gloves"
        ],
        "queries_neg": [
            "bare hands",
            "bare hand holding",
            "human hands bare",
            "bare human hands holding tools"
        ],
        "texts_pos": [
            "a close-up photo of hands wearing protective work gloves",
            "hands wearing dirty leather construction work gloves",
            "construction worker hands wearing heavy duty safety gloves",
            "protective work gloves worn on hands",
            "hands wearing safety work gloves holding tools",
            "industrial protective safety gloves on hands"
        ],
        "texts_neg": [
            "a close-up photo of bare hands without gloves",
            "bare hands without gloves holding a wheelbarrow handle or tools",
            "bare human hands skin without work gloves",
            "construction worker with bare exposed hands not wearing gloves",
            "bare skin on hands with fingers exposed without gloves"
        ]
    },
    "boot": {
        "body_part": "feet",
        "queries_pos": [
            "construction safety work boots",
            "steel toe work boots",
            "protective safety boots"
        ],
        "queries_neg": [
            "bare feet",
            "sneakers shoes feet",
            "casual sneakers without boots"
        ],
        "texts_pos": [
            "worker wearing heavy duty steel-toe safety work boots",
            "protective construction safety boots on feet",
            "industrial safety boots worn on construction site"
        ],
        "texts_neg": [
            "bare feet on ground",
            "casual sneakers shoes without safety boots",
            "person with ordinary sneakers not safety boots"
        ]
    },
    "goggle": {
        "body_part": "eyes",
        "queries_pos": [
            "safety goggles eye protection",
            "protective safety glasses construction",
            "safety glasses"
        ],
        "queries_neg": [
            "face without safety goggles",
            "human face bare eyes",
            "person face without glasses"
        ],
        "texts_pos": [
            "construction worker wearing protective safety goggles",
            "protective eye safety glasses worn on face",
            "safety goggles protecting eyes on construction site"
        ],
        "texts_neg": [
            "face without safety goggles or protective glasses",
            "worker face with bare exposed eyes without eye protection",
            "person face not wearing safety glasses"
        ]
    },
    "earplug": {
        "body_part": "ears",
        "queries_pos": [
            "hearing protection earmuffs",
            "safety earplugs industrial"
        ],
        "queries_neg": [
            "bare human ears without earplugs",
            "person ears"
        ],
        "texts_pos": [
            "worker wearing hearing protection earmuffs or safety earplugs"
        ],
        "texts_neg": [
            "worker with bare ears without hearing protection"
        ]
    }
}

class OpenVocab:
    """Items the detector was not trained on: Multi-source reference images (Wikimedia + Pixabay) + CLIP prototypes."""
    def __init__(s):
        s.m = None
        s.p = None
        s.protos = {}
        s.last_error = ""
        s.session = requests.Session()
        s.user_agent = getattr(C, "WIKIMEDIA_USER_AGENT", "SafetyApp/1.0 (d3208230@gmail.com)")
        s.wikimedia_endpoint = getattr(C, "WIKIMEDIA_ENDPOINT", "https://commons.wikimedia.org/w/api.php")
        s.session.headers.update({"User-Agent": s.user_agent})

    def _load(s):
        if s.m is None:
            s.m = CLIPModel.from_pretrained(NAME).eval()
            s.p = CLIPProcessor.from_pretrained(NAME)

    def embed(s, imgs):
        if not imgs:
            return torch.empty((0, 512))
        s._load()
        with C.LOCK, torch.no_grad():
            px = s.p(images=imgs, return_tensors="pt")["pixel_values"]
            return norm(s.m.visual_projection(s.m.vision_model(pixel_values=px).pooler_output), dim=-1)

    def embed_text(s, texts):
        s._load()
        with C.LOCK, torch.no_grad():
            inp = s.p(text=texts, return_tensors="pt", padding=True)
            return norm(s.m.text_projection(s.m.text_model(**inp).pooler_output), dim=-1)

    def search_wikimedia(s, q, n=6):
        """Search Wikimedia Commons for authentic real-world documentary and safety images."""
        try:
            params = {
                "action": "query",
                "format": "json",
                "generator": "search",
                "gsrsearch": q,
                "gsrnamespace": 6,  # File namespace
                "gsrlimit": n + 4,
                "prop": "imageinfo",
                "iiprop": "url|mime|size",
                "iiurlwidth": 400
            }
            res = s.session.get(s.wikimedia_endpoint, params=params, timeout=10)
            if res.status_code != 200:
                print(f"[OpenVocab] Wikimedia Commons HTTP {res.status_code} for '{q}'")
                return []
            data = res.json()
            pages = data.get("query", {}).get("pages", {})
        except Exception as e:
            print(f"[OpenVocab] Wikimedia Commons query failed for '{q}': {e}")
            return []

        out = []
        for pid, page in pages.items():
            info = page.get("imageinfo", [{}])[0]
            thumb = info.get("thumburl") or info.get("url")
            mime = info.get("mime", "").lower()
            # Filter for raster web image formats, skip SVGs, videos, and PDFs
            if not thumb or not any(fmt in mime for fmt in ("jpeg", "png", "webp", "jpg")):
                continue
            try:
                img_res = s.session.get(thumb, timeout=10)
                if img_res.status_code == 200 and len(img_res.content) > 100:
                    im = Image.open(io.BytesIO(img_res.content)).convert("RGB")
                    raw_title = page.get("title", "").replace("File:", "").split(".")[0][:30]
                    clean_title = raw_title.encode("ascii", "ignore").decode().replace("_", " ").strip()
                    if not clean_title:
                        clean_title = "Documentary Photo"
                    desc_url = info.get("descriptionurl") or thumb
                    out.append((im, dict(url=desc_url, user=f"Wikimedia Commons ({clean_title})")))
                    if len(out) >= n:
                        break
            except Exception:
                pass
        return out

    def search_pixabay(s, q, n=6):
        """Search Pixabay as secondary / backup image source."""
        key = os.getenv("PIXABAY_API_KEY")
        if not key:
            return []
        try:
            r = s.session.get("https://pixabay.com/api/", timeout=10, params=dict(
                key=key, q=q, image_type="photo", per_page=n + 4, safesearch="true")).json()
        except Exception as e:
            print(f"[OpenVocab] Pixabay query failed for '{q}': {e}")
            return []

        out = []
        for h in r.get("hits", [])[:n]:
            url = h.get("previewURL") or h.get("webformatURL")
            if not url:
                continue
            try:
                res = s.session.get(url, timeout=10)
                if res.status_code == 200 and len(res.content) > 100:
                    im = Image.open(io.BytesIO(res.content)).convert("RGB")
                    out.append((im, dict(url=h.get("pageURL", url), user=f"Pixabay ({h.get('user', 'Pixabay')})")))
            except Exception:
                pass
        return out

    def search_multi(s, queries, n=6):
        """Multi-source search: Wikimedia Commons primary, Pixabay backup."""
        collected = []
        # 1. Primary: Wikimedia Commons
        for q in queries:
            needed = n - len(collected)
            if needed <= 0:
                break
            wiki_hits = s.search_wikimedia(q, n=needed)
            collected.extend(wiki_hits)

        # 2. Backup: Pixabay if Wikimedia didn't return enough images
        if len(collected) < n:
            needed = n - len(collected)
            for q in queries:
                pix_hits = s.search_pixabay(q, n=needed)
                collected.extend(pix_hits)
                if len(collected) >= n:
                    break

        return collected[:n]

    def _mean(s, imgs):
        if not imgs:
            return None
        e = s.embed(imgs)
        if len(e) == 0:
            return None
        m = norm(e.mean(0), dim=0)
        keep = (e @ m).argsort(descending=True)[:max(2, int(len(imgs) * .7))]  # drop outlier references
        return norm(e[keep].mean(0), dim=0)

    def _get_spec(s, item):
        clean = item.lower().strip()
        # singular form mapping (gloves -> glove, boots -> boot, goggles -> goggle)
        base = clean.rstrip("s") if clean.endswith("s") and clean not in ("harness", "dress") else clean
        if base in PPE_SPECS:
            return PPE_SPECS[base]

        # Smart defaults for arbitrary open-vocab items
        return {
            "body_part": "torso",
            "queries_pos": [f"protective {clean}", f"worker wearing {clean}", f"safety {clean}"],
            "queries_neg": [f"person without {clean}", f"worker without {clean}", f"without {clean}"],
            "texts_pos": [
                f"a photo of a worker wearing {clean}",
                f"worker wearing protective safety {clean}",
                f"protective {clean} worn on person"
            ],
            "texts_neg": [
                f"a photo of a person without {clean}",
                f"worker not wearing {clean}",
                f"person bare without {clean}"
            ]
        }

    def _build(s, item):
        spec = s._get_spec(item)
        body_part = spec.get("body_part", "torso")

        # 1. Zero-shot CLIP text prototype vectors
        texts_pos = spec.get("texts_pos", [f"wearing {item}"])
        texts_neg = spec.get("texts_neg", [f"without {item}"])
        text_pos_v = norm(s.embed_text(texts_pos).mean(0), dim=0)
        text_neg_v = norm(s.embed_text(texts_neg).mean(0), dim=0)

        # 2. Reference image retrieval (Wikimedia primary, Pixabay backup)
        pos_imgs = s.search_multi(spec["queries_pos"], n=6)
        neg_imgs = s.search_multi(spec["queries_neg"], n=6)

        # 3. Multimodal ensemble if sufficient reference images are available
        if len(pos_imgs) >= 2 and len(neg_imgs) >= 2:
            p_img_mean = s._mean([x[0] for x in pos_imgs])
            n_img_mean = s._mean([x[0] for x in neg_imgs])
            if p_img_mean is not None and n_img_mean is not None:
                combo_pos = norm(0.5 * p_img_mean + 0.5 * text_pos_v, dim=0)
                combo_neg = norm(0.5 * n_img_mean + 0.5 * text_neg_v, dim=0)
                sources = [x[1] for x in pos_imgs[:3] + neg_imgs[:3]]
                sources.append(dict(url="https://openai.com/research/clip", user="CLIP Zero-Shot Ensemble"))
                s.protos[item] = dict(
                    t=time.time(),
                    pos=combo_pos,
                    neg=combo_neg,
                    body_part=body_part,
                    sources=sources
                )
                s.last_error = ""
                return s.protos[item]

        # 4. Pure CLIP zero-shot text fallback
        print(f"[OpenVocab] Fetched {len(pos_imgs)} pos / {len(neg_imgs)} neg reference images for '{item}'. Using CLIP zero-shot text prototype.")
        sources = [x[1] for x in pos_imgs + neg_imgs]
        sources.append(dict(url="https://openai.com/research/clip", user="CLIP Zero-Shot Text Embeddings"))
        s.protos[item] = dict(
            t=time.time(),
            pos=text_pos_v,
            neg=text_neg_v,
            body_part=body_part,
            sources=sources
        )
        s.last_error = ""
        return s.protos[item]

    def proto(s, item):
        c = s.protos.get(item)
        if c and time.time() - c["t"] < TTL:
            return c
        try:
            return s._build(item)
        except Exception as e:
            s.last_error = f"{type(e).__name__}: {e}"
            traceback.print_exc()
            spec = s._get_spec(item)
            texts_pos = spec.get("texts_pos", [f"wearing {item}"])
            texts_neg = spec.get("texts_neg", [f"without {item}"])
            pos_v = norm(s.embed_text(texts_pos).mean(0), dim=0)
            neg_v = norm(s.embed_text(texts_neg).mean(0), dim=0)
            s.protos[item] = dict(
                t=time.time(),
                pos=pos_v,
                neg=neg_v,
                body_part=spec.get("body_part", "torso"),
                sources=[dict(url="https://openai.com/research/clip", user="CLIP Zero-Shot Text Fallback")]
            )
            return s.protos[item]

    def person_state(s, img, box, item):
        pr = s.proto(item)
        if pr is None:
            return "uncertain", 0.0

        img_h, img_w = img.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)
        w, h = x2 - x1, y2 - y1

        if w < 16 or h < 24:
            return "uncertain", 0.0

        # Person partially cropped by the edge of the video/image frame
        is_edge = (x1 <= 0.02 * img_w or x2 >= 0.98 * img_w or y1 <= 0.02 * img_h or y2 >= 0.98 * img_h)

        body_part = pr.get("body_part", "torso")
        crops = []

        if body_part == "hands":
            # Multi-region hand cropping to isolate hands from bright vest / background
            crops = [
                img[y1 + int(.40 * h):y1 + int(.92 * h), x1:x1 + int(.55 * w)],                   # left arm/hand
                img[y1 + int(.40 * h):y1 + int(.92 * h), max(x1, x2 - int(.55 * w)):x2],         # right arm/hand
                img[y1 + int(.45 * h):y1 + int(.90 * h), x1 + int(.20 * w):max(x1 + int(.20 * w) + 10, x2 - int(.20 * w))], # center holding hands
                img[y1 + int(.48 * h):y1 + int(.92 * h), x1:x2],                                   # lower hand/arm band
                img[y1 + int(.30 * h):y1 + int(.85 * h), x1:x2]                                    # general upper body
            ]
        elif body_part == "feet":
            crops = [
                img[y1 + int(.70 * h):y2, x1:x2],                                                 # lower feet band
                img[y1 + int(.75 * h):y2, x1:x1 + int(.55 * w)],                                  # left foot
                img[y1 + int(.75 * h):y2, max(x1, x2 - int(.55 * w)):x2]                          # right foot
            ]
        elif body_part in ("eyes", "ears", "head"):
            crops = [
                img[y1:y1 + int(.40 * h), x1:x2],                                                 # head/face band
                img[y1 + int(.05 * h):y1 + int(.40 * h), x1 + int(.20 * w):max(x1 + int(.20 * w) + 10, x2 - int(.20 * w))] # center face
            ]
        else:
            crops = [
                img[y1 + int(.25 * h):y1 + int(.85 * h), x1:x2],                                   # torso band
                img[y1 + int(.15 * h):y1 + int(.85 * h), x1:x2]                                    # upper body
            ]

        valid_crops = [Image.fromarray(c[..., ::-1]) for c in crops if c.size > 0 and min(c.shape[:2]) >= 16]
        if not valid_crops:
            return "uncertain", 0.0

        embs = s.embed(valid_crops)
        diffs = [float(e @ pr["pos"] - e @ pr["neg"]) for e in embs]
        max_d = max(diffs)
        min_d = min(diffs)

        # Decision rule:
        # If any hand/body crop shows clear evidence of wearing the item -> compliant
        if max_d > MARGIN:
            return "compliant", min(1.0, max_d * 4)

        # If no crop shows compliance and bare skin / absence is detected:
        # If the worker is cut off at the edge of the screen, mark uncertain to prevent false boundary spikes
        if min_d < -MARGIN:
            if is_edge:
                return "uncertain", 0.0
            return "violation", min(1.0, abs(min_d) * 4)

        return "uncertain", 0.0
