import os, threading
os.environ.setdefault('YOLO_AUTOINSTALL', 'False')  # stop ultralytics pip-installing mid-run (it restarted the server)
LOCK = threading.RLock()
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "best.pt"
UP, EVID = ROOT / "data" / "uploads", ROOT / "data" / "evidence"
for d in (UP, EVID): d.mkdir(parents=True, exist_ok=True)
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
WIKIMEDIA_USER_AGENT = os.getenv("WIKIMEDIA_USER_AGENT", "SafetyApp/1.0 (d3208230@gmail.com)")
WIKIMEDIA_ENDPOINT = os.getenv("WIKIMEDIA_ENDPOINT", "https://commons.wikimedia.org/w/api.php")
CONF, PERSON_CONF, VIOL_CONF, STRIDE = 0.35, 0.40, 0.50, 1.0
