# VOXLENS — Voice Analytics Web Application

> Assignment 2 · Technical · Voice Analytics

A web application that ingests sales / support call recordings and produces:

- ✅ Full **transcript** with timestamps and speaker labels
- ✅ Conversation **summary** (abstractive)
- ✅ **Sentiment analysis** — overall, per-speaker, and trajectory across the call
- ✅ **Key Performance Indicators**
  - Customer Satisfaction (CSAT) score 0–100
  - Agent Performance score 0–100
  - Context-aware KPIs for **sales** (lead quality, buying intent, objections) and **support** (resolution signals, escalation, top concerns)
- ✅ Context selector — user picks **sales / support / general** before analysing

![VOXLENS UI sketch](docs/ui-sketch.svg)

---

## 🚀 Quick start

### Prerequisites
- Python **3.10+**
- `ffmpeg` available on your PATH (Whisper uses it for audio decoding)
  - macOS: `brew install ffmpeg`
  - Ubuntu: `sudo apt install ffmpeg`
  - Windows: download from <https://ffmpeg.org/download.html>

### Install and run

```bash
# 1. Clone your repo
git clone https://github.com/<your-username>/voxlens-voice-analytics.git
cd voxlens-voice-analytics

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install backend dependencies
pip install -r backend/requirements.txt

# 4. (Optional) Use OpenAI for higher-quality summaries
export OPENAI_API_KEY=sk-...

# 5. Launch
cd backend
python app.py
```

Open <http://localhost:5000> and upload a call recording.

> First run downloads the Whisper and distilbart models (~600 MB total). Subsequent runs are instant.

---

## 📂 Project structure

```
voxlens-voice-analytics/
├── backend/
│   ├── app.py              # Flask routes + file handling
│   ├── analyzer.py         # 5-stage analysis pipeline
│   ├── requirements.txt
│   ├── uploads/            # Saved audio (gitignored)
│   └── results/            # Saved JSON results (gitignored)
├── frontend/
│   ├── index.html          # Single-page app
│   ├── styles.css          # Editorial / lab aesthetic
│   └── app.js              # Upload + render logic
├── samples/
│   ├── sample_input_support.txt
│   └── sample_output_support.json
├── docs/
│   └── ARCHITECTURE.md
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🧠 How it works (1-minute version)

| Stage | Tool | Output |
|------|------|--------|
| 1. Transcription | `faster-whisper` (base, int8) | Timestamped text |
| 2. Speaker tagging | Rule-based heuristic | Agent / Customer labels |
| 3. Summarisation | `distilbart-cnn-12-6` (local) **or** `gpt-4o-mini` (if `OPENAI_API_KEY` set) | 4–6 bullet summary |
| 4. Sentiment | VADER | Overall + per-speaker + trajectory |
| 5. KPIs | Lexicon + sentiment composition | CSAT, agent performance, context KPIs |

Full details: see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## 🛠️ Technologies used

**Backend**
- Python 3.10+
- Flask · Flask-CORS
- faster-whisper (CTranslate2-accelerated OpenAI Whisper)
- transformers + PyTorch (for distilbart summariser)
- vaderSentiment

**Frontend**
- Vanilla HTML / CSS / JavaScript (no build step)
- Google Fonts: *Fraunces*, *JetBrains Mono*, *Inter*
- Single-page app; uses `fetch` + `FormData`

**Why no React / Vue?** Keeping the frontend dependency-free means the entire project boots with one `python app.py` command — no `npm install`, no bundler, no node_modules. The backend serves the static files directly.

---

## 🌐 API reference

### `POST /api/analyze`
**Body** (multipart/form-data):
- `audio` — the audio file (mp3 / wav / m4a / ogg / flac / webm)
- `context` — `"sales"` | `"support"` | `"general"`

**Returns** `application/json` matching `samples/sample_output_support.json`.

### `GET /api/result/<job_id>`
Retrieve a previously computed result by its ID.

### `GET /api/health`
Liveness check.

---

## 📊 Sample input & output

A complete worked example lives in [`samples/`](samples/):
- `sample_input_support.txt` — what a 3-minute support call looks like
- `sample_output_support.json` — the full JSON the API returns

Selected highlights from the sample:

```
CSAT score:           78 / 100   (High)
Agent score:          84 / 100   (Strong)
Overall sentiment:    +0.48      (Positive)
Sentiment arc:        Neutral → Neutral → Positive
Likely resolved:      YES
Escalation:           No
```

---

## 🧪 Testing without uploading audio

If you don't have an audio file handy, you can verify the pipeline by:

```bash
# Use a Creative-Commons sample call (any short mp3 will do)
curl -F "audio=@/path/to/short_clip.mp3" \
     -F "context=support" \
     http://localhost:5000/api/analyze | python -m json.tool
```

---

## 🚧 Known limitations

- Speaker tagging is heuristic (alternating turns). True diarization would require `pyannote.audio` + a Hugging Face token.
- English-only by default; Whisper itself is multilingual but the lexicons and sentiment model are tuned for English.
- The free `base` Whisper model trades a little accuracy for speed. Swap to `small` or `medium` in `analyzer.py` for higher fidelity.

---

## 📄 License

MIT — see `LICENSE`.
