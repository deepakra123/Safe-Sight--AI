# Run from this folder with the venv active:  python diag.py
import os
from app import config as C
key = os.getenv("PIXABAY_API_KEY")
print("PIXABAY key loaded:", bool(key), "| length:", len(key or ""))
print("GROQ key loaded:", bool(os.getenv("GROQ_API_KEY")))
from app.cv.openvocab import OpenVocab
ov = OpenVocab()
pos, neg = ov.search("worker wearing gloves"), ov.search("worker without gloves bare")
print("downloaded:", len(pos), len(neg))
e = ov.embed([x[0] for x in pos]); print("CLIP embeddings:", tuple(e.shape))
print("prototype built:", ov._build("gloves") is not None)
