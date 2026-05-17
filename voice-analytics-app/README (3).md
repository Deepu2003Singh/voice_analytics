# VoxLens — Voice Analytics Web App

A simple web app that takes a call recording (sales or support) and tells you what happened on the call — the transcript, a short summary, how the customer felt, and a few useful scores.

I built this as part of the Burger Singh pre-joining assignment.

## What it does

You upload an audio file and pick whether it's a sales call, a support call, or just a general one. The app then gives you back:

- The full **transcript** with timestamps, and a guess at who is the agent and who is the customer
- A short **summary** in 4–6 bullet points so you don't have to read the whole transcript
- **Sentiment** — was the call positive, negative or neutral? And how did the mood change from start to end?
- A few **scores out of 100**:
  - CSAT (how happy the customer seems)
  - Agent performance (did the agent handle things well?)
- **Extra KPIs based on context** — for sales calls it tells you if the lead is hot/warm/cold, for support calls it tells you if the issue got resolved or if the customer asked to escalate

There is also a small UI sketch in `docs/ui-sketch.svg` if you want to see what it looks like.

## How to run it on your machine

You will need:

- Python 3.10 or newer
- `ffmpeg` installed (Whisper needs this to read audio files)
  - On Mac: `brew install ffmpeg`
  - On Ubuntu/Debian: `sudo apt install ffmpeg`
  - On Windows: download from https://ffmpeg.org/download.html and add to PATH

Then:

```bash
# 1. Clone the repo
git clone https://github.com/Deepu2003Singh/voice_analytics.git
cd voice_analytics
cd voice-analytics-app

# 2. Make a virtual environment (good practice, keeps things clean)
python3 -m venv .venv
source .venv/bin/activate
# On Windows use: .venv\Scripts\activate

# 3. Install the Python packages
pip install -r backend/requirements.txt

# 4. (Optional) If you have an OpenAI key, the summaries come out better
export OPENAI_API_KEY=sk-...

# 5. Start the server
cd backend
python app.py
```

Now open http://localhost:5000 in your browser and upload a call.

**Heads up:** the first time you run it, the app downloads the Whisper and summarizer models. That's around 600 MB so it takes a few minutes. After that, it's quick.

## Folder layout

```
voxlens-voice-analytics/
├── backend/
│   ├── app.py              # Flask server, handles uploads and routes
│   ├── analyzer.py         # The actual analysis pipeline (5 stages)
│   ├── requirements.txt
│   ├── uploads/            # Where uploaded audio gets saved (ignored by git)
│   └── results/            # Where the JSON results get saved (ignored by git)
├── frontend/
│   ├── index.html          # The single page UI
│   ├── styles.css          # Styling
│   └── app.js              # Upload handling and rendering the results
├── samples/
│   ├── sample_input_support.txt
│   └── sample_output_support.json
├── docs/
│   └── ARCHITECTURE.md     # Longer write-up of how the system works
├── .gitignore
├── LICENSE
└── README.md
```

## How it works (short version)

The audio file goes through 5 steps:

| Step | What it does | Tool used |
|------|-------------|-----------|
| 1 | Convert audio to text | `faster-whisper` (a fast version of OpenAI's Whisper) |
| 2 | Figure out who is talking (agent vs customer) | Simple rule-based logic |
| 3 | Make a short summary | `distilbart-cnn-12-6` locally, OR `gpt-4o-mini` if you set an API key |
| 4 | Check the mood of the call | VADER sentiment library |
| 5 | Calculate the KPI scores | Mix of keyword lists and sentiment numbers |

For the longer version of all this, check out [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Tech stack

**Backend**
- Python 3.10+
- Flask + Flask-CORS for the web server
- faster-whisper for speech-to-text
- transformers + PyTorch for summarization
- vaderSentiment for sentiment analysis

**Frontend**
- Plain HTML, CSS, and JavaScript — no React, no Vue, no build tools
- Fonts from Google Fonts (Fraunces, JetBrains Mono, Inter)

I deliberately kept the frontend simple. No `npm install`, no bundlers, no node_modules folder. The Flask backend just serves the static files directly, so to run the whole project you only need to run one command: `python app.py`. That felt cleaner for an assignment.

## API endpoints

If you want to call the API directly instead of using the UI:

### `POST /api/analyze`
Send a multipart form with:
- `audio` — the audio file (mp3, wav, m4a, ogg, flac, or webm)
- `context` — one of `"sales"`, `"support"`, or `"general"`

You get back a JSON response. There's a full example in `samples/sample_output_support.json`.

### `GET /api/result/<job_id>`
Get an earlier result back if you have the job ID.

### `GET /api/health`
Just checks if the server is alive.

## Sample input and output

I included a full worked example in the `samples/` folder so you can see the kind of output the system produces without having to upload anything yourself:

- `sample_input_support.txt` — the transcript of a 3-minute support call
- `sample_output_support.json` — the full JSON response the API gave for that call

Some of the things you can see in the sample output:

```
CSAT score:           78 / 100   (High)
Agent score:          84 / 100   (Strong)
Overall sentiment:    +0.48      (Positive)
Sentiment arc:        Neutral → Neutral → Positive
Likely resolved:      YES
Escalation:           No
```

## Testing without an audio file

If you don't have a call recording handy but want to make sure the pipeline runs, you can use any short mp3:

```bash
curl -F "audio=@/path/to/short_clip.mp3" \
     -F "context=support" \
     http://localhost:5000/api/analyze | python -m json.tool
```

Any creative commons audio clip from the internet will work.

## Things that could be better

I want to be honest about the limitations so you know what's good and what isn't:

- **Speaker tagging is basic.** I just assume the agent and customer take turns. Real diarization (figuring out who's talking from the audio itself) needs `pyannote.audio` and a Hugging Face token, which felt like too much setup for an assignment. The code is structured so this can be swapped out easily later.
- **English only.** Whisper itself supports many languages, but my sentiment library and keyword lists are tuned for English.
- **The Whisper `base` model is fast but not the most accurate.** If you need better accuracy, change `base` to `small` or `medium` in `analyzer.py` — it'll be slower but more accurate.

## License

MIT — see the `LICENSE` file.

## A note on AI tools used

I used Claude (Anthropic) to help me with the architecture decisions, write parts of the analyzer code, and clean up the frontend styling. I used ChatGPT for debugging when I got stuck on the Whisper integration. I reviewed and tested everything myself, and changed quite a bit of the AI-suggested code — especially the KPI scoring logic, which I rewrote a few times to make the numbers feel reasonable on the sample calls I tested with.
