# Architecture & System Logic

This document explains how **VOXLENS** processes a call recording end-to-end.

---

## 1. High-level architecture

```
 ┌─────────────────────┐     HTTP/multipart      ┌──────────────────────────┐
 │  Browser frontend   │  ─────────────────────► │  Flask backend (app.py)  │
 │  (HTML/CSS/JS SPA)  │ ◄───────────────────── │                          │
 └─────────────────────┘     JSON response        └──────────────┬───────────┘
                                                                  │
                                                                  ▼
                                                  ┌──────────────────────────┐
                                                  │  VoiceAnalyzer pipeline  │
                                                  │  (analyzer.py)           │
                                                  └──────────────┬───────────┘
                                                                  │
                ┌──────────────┬───────────────┬─────────────────┼─────────────────┐
                ▼              ▼               ▼                 ▼                 ▼
         Whisper ASR     Speaker tagging   Summariser       VADER sentiment    KPI engine
         (faster-       (rule-based       (distilbart      (lexicon +         (rule-based +
          whisper)        heuristic)       OR OpenAI)       compound score)    lexicons)
```

The browser uploads an audio file to the Flask backend. The backend persists
the file to disk and hands the path to the **VoiceAnalyzer**, which runs the
five-stage pipeline and returns a single JSON payload that drives the
dashboard.

---

## 2. Pipeline stages

### Stage 1 — Transcription
- Library: **`faster-whisper`** running the `base` model in CPU `int8` mode.
- Audio is decoded and run through Whisper with VAD (voice-activity-detection)
  filtering so silent regions don't generate hallucinated text.
- Output: full transcript string + a list of `{start, end, text}` segments.

### Stage 2 — Speaker-role tagging
- True speaker diarization (pyannote) requires a Hugging Face token and adds
  hundreds of MB of model weight. For this assignment we use a **lightweight
  heuristic**:
  - If the first segment contains a greeting like *"thank you for calling…"*,
    *"how can I help…"*, *"this is X speaking"*, the first speaker is tagged
    **Agent**, otherwise **Customer**.
  - Subsequent turns alternate Agent ↔ Customer.
- This is intentionally simple and is the cleanest place in the codebase to
  upgrade with real diarization later — `analyzer.py::_label_speakers` is the
  only function that needs to change.

### Stage 3 — Summarisation
Two-tier strategy:
1. If `OPENAI_API_KEY` is set in the environment, the transcript is sent to
   `gpt-4o-mini` with a prompt designed for call summaries.
2. Otherwise, the local Hugging Face pipeline
   `sshleifer/distilbart-cnn-12-6` produces an abstractive summary. Long
   transcripts are chunked (700 words / chunk) and the chunk summaries are
   concatenated.
3. If both fail, an **extractive fallback** scores each sentence by sum of
   word frequencies (a tiny TextRank-like heuristic) and keeps the top 5.

### Stage 4 — Sentiment analysis
- Library: **VADER** (`vaderSentiment`) — a lexicon + rules model tuned for
  conversational text. It returns four scores per snippet: `pos`, `neu`,
  `neg`, `compound` (the `-1 … +1` aggregate).
- We compute:
  - **Overall** sentiment for the whole transcript.
  - **Per-speaker** average compound score (Agent vs Customer).
  - **Trajectory** — segments are split into thirds (Start / Middle / End)
    and each third's average compound score is reported. This captures the
    arc of the call ("started frustrated, ended happy").

### Stage 5 — KPI extraction
KPIs are computed from a combination of lexicons and sentiment scores.

**Customer satisfaction (CSAT, 0–100)**
```
csat = clamp( 50 + (customer_compound * 50) + 3*(positive_cues − negative_cues) , 0, 100 )
```
- `positive_cues`: occurrences of phrases such as *"thank you"*, *"great"*,
  *"perfect"*, *"appreciate"*, etc.
- `negative_cues`: occurrences of *"frustrated"*, *"refund"*, *"useless"*, etc.

**Agent performance (0–100)**
```
agent = 50 + 5*good_behaviours − 10*poor_behaviours − 0.5*filler_words
        + 20*agent_compound − 8*(if talk-ratio outside [0.3, 0.7])
```
- `good_behaviours`: *"let me check"*, *"I understand"*, *"happy to"*, *"just
  to confirm"*, etc.
- `poor_behaviours`: *"I don't know"*, *"not my problem"*, *"calm down"*,
  *"policy is policy"*, etc.
- `filler_words`: *um*, *uh*, *like*, *you know*, *basically*…
- `talk_ratio` = `agent_words / total_words`. A healthy agent listens roughly
  as much as they speak; we penalise dominance or absence.

**Context-specific KPIs**

| Context | KPIs produced |
| ------- | ------------- |
| **Sales** | buying-intent signal count, objection signal count, commitment-secured (bool), lead quality (Hot/Warm/Cold) |
| **Support** | resolution signal count, escalation requested (bool), likely-resolved (bool), top customer concerns (keywords) |
| **General** | conversation metrics only |

**Conversation metrics** (always emitted)
- duration (seconds / minutes)
- turn count
- per-speaker word count
- words per minute
- top 10 keywords (stop-word-filtered frequency count)

---

## 3. Why these choices?

| Decision | Rationale |
| --- | --- |
| `faster-whisper` over OpenAI Whisper API | Runs locally, no API key, free; `base` model is accurate enough for English calls and finishes a 3-minute clip in ~10–20s on CPU. |
| Heuristic speaker tagging | Avoids the ~500 MB pyannote download and HF auth flow. Sufficient for the assignment; clearly isolated for future upgrade. |
| VADER over a fine-tuned transformer | VADER is designed for conversational text, handles negation/intensifiers/emojis, and is essentially zero-latency. A BERT-based classifier would be slightly more accurate but 100× slower for marginal gain on a 5-minute call. |
| `distilbart-cnn-12-6` for summarisation | One of the smallest abstractive summarisers that still produces fluent English. Optionally switched to GPT-4o-mini if the user provides an API key. |
| Lexicon-based KPIs over ML scoring | Transparent, auditable, no labelled training data needed, and gives reasonable scores on the assignment scope. The system is structured so each KPI can be swapped to an ML model behind the same interface. |
| Flask | Simplest Python web stack that ships a JSON API and serves the static frontend in the same process. Easy to containerise and deploy. |

---

## 4. Request / response lifecycle

```
1. User selects audio + context, clicks "Run analysis".
2. Browser POSTs multipart form to /api/analyze.
3. Backend:
   a. Validates extension and size (≤50 MB).
   b. Saves upload to /backend/uploads/<job_id>.<ext>.
   c. Runs VoiceAnalyzer.analyze(path, context).
   d. Persists the result to /backend/results/<job_id>.json.
   e. Returns the result JSON.
4. Frontend hides the loading state and renders:
   - KPI cards (CSAT / Agent / Sentiment)
   - Summary bullets
   - Sentiment trajectory
   - Context-specific KPI grid
   - Keyword chips
   - Turn-by-turn transcript with per-turn sentiment
5. User can copy transcript, download JSON, or start a new analysis.
```

---

## 5. Failure modes & handling

| Failure | Behaviour |
| --- | --- |
| Invalid file type | Backend returns 400; UI shows red error state. |
| File > 50 MB | Werkzeug rejects with 413; UI surfaces the message. |
| OpenAI key set but request fails | Auto-falls back to local distilbart. |
| distilbart fails (e.g. out of memory) | Falls back to extractive summary. |
| Whisper produces empty transcript | Pipeline returns `transcript: ""` and degrades gracefully — summary becomes empty, sentiment is neutral, KPIs reflect 0 turns. |

---

## 6. Extension points

- **Diarization** — replace `_label_speakers` with `pyannote.audio` for
  multi-speaker calls.
- **Emotion** — augment VADER with a fine-tuned RoBERTa-emotion classifier.
- **PII redaction** — pre-process the transcript with `presidio` or regex
  patterns to scrub names, emails, card numbers.
- **Storage** — swap the file-based `results/` folder for SQLite or
  Postgres. The result JSON shape is already flat enough to insert.
- **Auth** — add Flask-Login for multi-tenant use.
