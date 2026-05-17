"""
Voice Analytics Web Application — Backend API
==============================================
A Flask service that ingests call recordings (sales/support) and returns:
  • Full transcript
  • Conversation summary
  • Sentiment analysis (overall + segment-level)
  • Key Performance Indicators (customer satisfaction, agent performance)

Author: <your name>
License: MIT
"""

import os
import uuid
import json
import logging
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

from analyzer import VoiceAnalyzer

# ------------------------------------------------------------------ #
#  App setup                                                         #
# ------------------------------------------------------------------ #
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {"mp3", "wav", "m4a", "ogg", "flac", "webm", "mp4"}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB

app = Flask(__name__, static_folder=str(BASE_DIR.parent / "frontend"), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
CORS(app)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("voice-analytics")

analyzer = VoiceAnalyzer()


# ------------------------------------------------------------------ #
#  Helpers                                                           #
# ------------------------------------------------------------------ #
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ------------------------------------------------------------------ #
#  Routes                                                            #
# ------------------------------------------------------------------ #
@app.route("/")
def index():
    """Serve the frontend single-page app."""
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "timestamp": datetime.utcnow().isoformat()})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Main analysis endpoint.

    Form fields:
      • audio   – the uploaded audio file
      • context – "sales" | "support" | "general"
    """
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    f = request.files["audio"]
    if f.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    if not allowed_file(f.filename):
        return jsonify({"error": f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    context = request.form.get("context", "general").lower()
    if context not in {"sales", "support", "general"}:
        context = "general"

    # Save upload
    job_id = uuid.uuid4().hex[:12]
    ext = f.filename.rsplit(".", 1)[1].lower()
    safe_name = f"{job_id}.{ext}"
    saved_path = UPLOAD_DIR / safe_name
    f.save(saved_path)
    log.info("Saved upload %s (context=%s)", saved_path, context)

    try:
        result = analyzer.analyze(str(saved_path), context=context)
        result["job_id"] = job_id
        result["context"] = context
        result["filename"] = secure_filename(f.filename)
        result["analyzed_at"] = datetime.utcnow().isoformat()

        # Persist result
        with open(RESULTS_DIR / f"{job_id}.json", "w", encoding="utf-8") as fp:
            json.dump(result, fp, indent=2)

        return jsonify(result)
    except Exception as e:
        log.exception("Analysis failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/result/<job_id>")
def get_result(job_id: str):
    path = RESULTS_DIR / f"{job_id}.json"
    if not path.exists():
        return jsonify({"error": "Not found"}), 404
    with open(path, encoding="utf-8") as fp:
        return jsonify(json.load(fp))


# ------------------------------------------------------------------ #
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
