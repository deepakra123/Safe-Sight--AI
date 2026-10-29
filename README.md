# Safe-Sight--AI

SafeSight AI is an AI-powered safety monitoring system that uses computer vision to identify potential PPE and workplace safety risks.

## PPE video/image Q&A

1. `python -m venv venv` then activate it, then `pip install -r requirements.txt`
2. Put your trained detector at `models/best.pt`; copy `.env.example` to `.env` and add the keys.
3. Run from this folder: `uvicorn app.server.main:app --reload --reload-dir app --reload-dir frontend` and open http://localhost:8000

Flow: upload -> background scan (hardhat, vest) -> ask. Items outside hardhat/vest/mask (e.g. gloves) use Pixabay reference images + CLIP (first use downloads CLIP and fetches references; images are kept in memory only).
