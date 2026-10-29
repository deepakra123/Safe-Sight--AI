import json, re
from typing import Optional, Literal, TypedDict
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from .. import config as C
from ..cv.engine import engine, ITEMS

class Intent(BaseModel):
    in_scope: bool            # is this a question about PPE / safety in the uploaded media?
    item: str = ""            # hardhat | vest | mask | or a singular noun such as gloves, goggles, boots
    qtype: Literal["yesno", "when", "count", "show", "summary"] = "yesno"
    t_start: Optional[float] = None   # seconds
    t_end: Optional[float] = None
    wants_evidence: bool = False

class Answer(BaseModel):
    answer: str
    verdict: Literal["yes", "no", "partial", "cannot_determine"]

class S(TypedDict, total=False):
    failed: bool; media_id: str; question: str; intent: Intent; facts: dict; evidence: list; result: dict

_llm = None
def llm():
    global _llm
    if _llm is None: _llm = ChatGroq(model=C.GROQ_MODEL, temperature=0)
    return _llm

EXTRA = ["glove", "goggle", "boot", "glasses", "earplug", "harness"]
SYN = {"hardhat": ["helmet", "hard hat", "hardhat"], "vest": ["vest", "hi-vis", "high vis"], "mask": ["mask"]}
def parse(st):
    q = st["question"]
    try:
        i = llm().with_structured_output(Intent).invoke([("system",
            "Parse a question about PPE in an uploaded image/video. Map helmet/hard hat->hardhat, safety vest/hi-vis->vest, "
            "mask->mask; other items stay as a singular lowercase noun (gloves, goggles). Times in seconds. "
            "in_scope=false if it is not about PPE/safety."), ("user", q)])
    except Exception:
        item = next((k for k, v in SYN.items() if any(w in q.lower() for w in v)), "") or next((w for w in EXTRA if w in q.lower()), "")
        i = Intent(in_scope=bool(item), item=item)
    i.item = i.item.strip().lower()
    return {"intent": i}

def refuse(st):
    return {"result": dict(answer="I can only answer PPE questions about the uploaded image or video "
                                  "(for example: hardhats, vests, masks, gloves).", verdict="cannot_determine",
                           item="", citations=[], evidence=[], sources=[], coverage={})}

def gather(st):
    i, m = st["intent"], engine.get(st["media_id"])
    if i.item not in ITEMS and engine.ov.proto(i.item) is None:
        return {"failed": True, "result": dict(answer=f"I couldn't build reference examples for '{i.item}' ({engine.ov.last_error or 'unknown error'}), so I can't check it reliably. Hardhat, vest and mask always work.", verdict="cannot_determine", item=i.item, citations=[], evidence=[], sources=[], coverage={})}
    t = engine.tag(m, i.item)
    lo, hi = i.t_start if i.t_start is not None else 0, i.t_end if i.t_end is not None else m.dur + 1
    ev = [e for e in t["events"] if e["end_s"] > lo and e["start_s"] < hi]
    v, u = [e for e in ev if e["state"] == "violation"], [e for e in ev if e["state"] == "uncertain"]
    frames, seen = [], set()
    def add(sec, label):
        k = int(round(sec * m.fps))
        if k not in seen and len(frames) < 6:
            seen.add(k); n = engine.annotate(m, sec, i.item, label)
            frames.append(dict(url="/evidence/" + n, t=round(sec, 2), caption=f"{i.item} {label} at {k / m.fps:.1f} s"))
    for e in v[:3]:
        add(e["start_s"], "violation start"); add((e["start_s"] + e["end_s"]) / 2, "violation middle")
        add(max(e["start_s"], e["end_s"] - 1 / m.fps), "violation end")
    if not frames:
        e = u[0] if u else None
        add((e["start_s"] + e["end_s"]) / 2 if e else (lo + min(hi, m.dur)) / 2, "uncertain" if e else "sample frame")
    pr = engine.ov.protos.get(i.item)
    facts = dict(item=i.item, detector="trained" if i.item in ITEMS else "reference images + CLIP",
                 media="image" if m.is_image else "video", duration_s=round(m.dur, 2), question_range=[i.t_start, i.t_end],
                 violations=v, uncertain=u, coverage_percent=t["coverage"], frames_analysed=t["frames"])
    return {"facts": facts, "evidence": frames, "result": dict(sources=pr["sources"] if pr else [])}

def answer(st):
    f = st["facts"]
    try:
        a = llm().with_structured_output(Answer).invoke([("system",
            "Answer the question using ONLY these analysis results. 'violations' = intervals where the item is missing; "
            "'uncertain' = item not visible/unclear. Give times in seconds like '9.7 s'. Never invent times. verdict: yes/no = the "
            "asked condition is true/false; use partial or cannot_determine when evidence is only uncertain or mixed. "
            "Mention uncertainty briefly. If detector is 'reference images + CLIP', say it is a lower-confidence estimate.\n"
            + json.dumps(f)), ("user", st["question"])])
        return {"result": {**st["result"], "answer": a.answer, "verdict": a.verdict}}
    except Exception as ex:
        return {"result": {**st["result"], "answer": "", "verdict": "cannot_determine"}}

def template(f):
    v, u = f["violations"], f["uncertain"]
    if v: return "yes", f"{f['item']} missing in {len(v)} interval(s): " + ", ".join(f"{e['start_s']:.1f}-{e['end_s']:.1f} s" for e in v) + "."
    if u: return "cannot_determine", f"No confirmed violation, but {len(u)} segment(s) were unclear (item not visible)."
    return "no", f"No {f['item']} violation found."

def verify(st):
    f, r, i = st["facts"], dict(st["result"]), st["intent"]
    ok = [x for e in f["violations"] + f["uncertain"] for x in (e["start_s"], e["end_s"])] + [0, f["duration_s"], i.t_start or 0, i.t_end or 0]
    cited = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*s\b", r["answer"])]
    if not r["answer"] or any(not any(abs(c - x) <= .6 for x in ok) for c in cited):   # reject invented timestamps
        r["verdict"], r["answer"] = template(f)
    r.update(item=f["item"], citations=f["violations"] + f["uncertain"], evidence=st["evidence"], coverage=f["coverage_percent"])
    return {"result": r}

g = StateGraph(S)
for n, fn in [("parse", parse), ("refuse", refuse), ("gather", gather), ("answer", answer), ("verify", verify)]: g.add_node(n, fn)
g.set_entry_point("parse")
g.add_conditional_edges("parse", lambda s: "go" if s["intent"].in_scope and s["intent"].item else "stop", {"go": "gather", "stop": "refuse"})
g.add_conditional_edges("gather", lambda s: "stop" if s.get("failed") else "go", {"go": "answer", "stop": END}); g.add_edge("answer", "verify"); g.add_edge("verify", END); g.add_edge("refuse", END)
graph = g.compile()

def ask(media_id, question):
    return graph.invoke({"media_id": media_id, "question": question})["result"]
