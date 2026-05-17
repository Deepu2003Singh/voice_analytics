# Architecture & How the System Works

This document explains how VoxLens takes a call recording and turns it into useful insights, step by step. I wrote this for anyone who wants to understand the inner working without reading through all the code.

## 1. The big picture

Here's how the pieces fit together:

```
 ┌─────────────────────┐     HTTP/multipart      ┌──────────────────────────┐
 │  Browser frontend   │  ─────────────────────► │  Flask backend (app.py)  │
 │  (HTML/CSS/JS)      │ ◄───────────────────── │                          │
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
          whisper)        heuristic)       OR OpenAI)       compound score)    keyword lists)
```

In plain English: the browser sends the audio file to a Flask server. The server saves the file and runs it through the `VoiceAnalyzer`, which has 5 stages. Once all 5 stages are done, it sends back one big JSON object that the frontend uses to draw the dashboard.

## 2. The 5 stages explained

### Stage 1 — Turning audio into text (Transcription)

- I used **`faster-whisper`** with the `base` model in `int8` mode. This is basically a faster, lighter version of OpenAI's Whisper that runs on a normal CPU without needing a GPU.
- I also turned on VAD (Voice Activity Detection) filtering. This makes Whisper skip the silent parts of the audio, which is important because otherwise Whisper sometimes "hallucinates" text in silent gaps (like making up random words).
- The output is the full text of the call, plus a list of segments with start and end timestamps.

### Stage 2 — Figuring out who is talking

This is one place where I had to make a tradeoff for the assignment. Real speaker diarization (where the model figures out from the audio itself who's talking) needs `pyannote.audio`, which is a few hundred MB of models and needs a Hugging Face account and token. That felt like too much friction for an assignment.

So instead, I went with a simple rule:

- Look at the first thing said in the call.
- If it sounds like an agent's greeting — phrases like "thank you for calling", "how can I help you", "this is so-and-so speaking" — then the first speaker is the **Agent**.
- Otherwise, the first speaker is the **Customer**.
- After that, just assume the speakers alternate (Agent, Customer, Agent, Customer...).

It's a simplification but it works fine for one-on-one sales/support calls, which is the assignment scope. If someone wanted to make this more accurate later, the only function they'd need to replace is `_label_speakers` in `analyzer.py`. I deliberately kept it isolated for this reason.

### Stage 3 — Summarizing the conversation

I made the summary feature have 3 fallback levels, so it always returns something useful even if things fail:

1. **First choice:** If you set the `OPENAI_API_KEY` environment variable, the transcript goes to `gpt-4o-mini` with a prompt I wrote specifically for summarizing customer calls. This gives the best summaries.
2. **Second choice:** If no API key is set (or the API call fails), it falls back to the local model `sshleifer/distilbart-cnn-12-6` which runs on your own machine. For long calls, I chunk the transcript into 700-word pieces, summarize each piece, then join the summaries together.
3. **Last resort:** If even the local model fails (maybe out of memory), I use a basic extractive summary — basically just scoring each sentence by how often its words appear in the call, and picking the top 5 sentences. It's not great but it's better than nothing.

### Stage 4 — Sentiment analysis (how did the call feel?)

For sentiment I used **VADER** (the `vaderSentiment` library). I chose it over fancier deep learning models for a few reasons:

- It's specifically designed for conversational text
- It handles negation properly ("not bad" doesn't score as negative)
- It handles intensifiers ("really good" scores higher than "good")
- It's basically instant — no waiting for a model to run

VADER gives 4 scores for any piece of text: `pos`, `neu`, `neg`, and `compound` (which is the overall score from -1 to +1).

I compute three sentiment views:

- **Overall** — sentiment for the whole call
- **Per-speaker** — average score for the agent vs the customer separately. This is useful because sometimes the agent stays positive while the customer gets frustrated.
- **Trajectory** — I split the call into three parts (Start, Middle, End) and show how the mood changed. This catches the arc of the call — for example, "frustrated → neutral → positive" tells you the agent turned things around.

### Stage 5 — Calculating the KPI scores

This is where it gets a bit math-heavy. KPI scores come from a mix of keyword counting and sentiment numbers.

**Customer Satisfaction (CSAT) score, 0 to 100:**

```
csat = clamp( 50 + (customer_compound * 50) + 3*(positive_cues − negative_cues) , 0, 100 )
```

The idea is: start at 50 (neutral), then push it up or down based on:
- How positive or negative the customer sounded overall (the compound score)
- How many positive things the customer said ("thank you", "great", "perfect", "appreciate it"...)
- How many negative things they said ("frustrated", "useless", "refund", "complaint"...)

The `clamp` just makes sure the final number stays between 0 and 100.

**Agent Performance score, 0 to 100:**

```
agent = 50 + 5*good_behaviours − 10*poor_behaviours − 0.5*filler_words
        + 20*agent_compound − 8*(if talk-ratio outside [0.3, 0.7])
```

Again, start at 50 and adjust based on:
- **Good behaviours** like "let me check that for you", "I understand", "happy to help", "just to confirm" → +5 each
- **Poor behaviours** like "I don't know", "not my problem", "calm down", "policy is policy" → -10 each (weighted more heavily because these are red flags)
- **Filler words** like "um", "uh", "like", "you know" → small penalty of 0.5 each
- **Agent's own sentiment** — agents should sound positive and helpful
- **Talk ratio penalty** — if the agent is talking less than 30% or more than 70% of the time, that's a sign of either not engaging or dominating the conversation, so -8

**Context-specific KPIs**

Depending on whether the call is sales, support, or general, you also get different KPIs:

| Context | What you get |
| --- | --- |
| **Sales** | How many buying-intent signals, how many objections, did the agent secure a commitment (yes/no), overall lead quality (Hot/Warm/Cold) |
| **Support** | How many resolution signals were said, did the customer ask to escalate (yes/no), is the issue likely resolved (yes/no), top customer concerns (keywords) |
| **General** | Just basic conversation metrics |

**Basic metrics** — these come out for every call regardless of context:

- How long the call was (in seconds and minutes)
- How many back-and-forth turns there were
- Word count for each speaker
- Words per minute (pace)
- Top 10 keywords from the call (after filtering out filler words like "the", "and", "is")

## 3. Why I made the choices I did

This is probably the most useful section if you're trying to understand the design decisions. Here's my reasoning for each main choice:

| Decision | Why |
| --- | --- |
| `faster-whisper` instead of OpenAI Whisper API | Runs locally so no API key is needed, costs nothing, and the `base` model is good enough for English calls. A 3-minute clip takes about 10–20 seconds on a normal CPU. |
| Simple rule-based speaker tagging | Avoids the 500 MB pyannote download and the Hugging Face token setup. It's fine for two-person calls (the assignment scope). And it's the cleanest part of the code to upgrade later if needed. |
| VADER instead of a fine-tuned transformer | VADER is built specifically for conversational text and handles negations, intensifiers, and informal language well. A BERT-based model would be slightly more accurate but 100x slower, which isn't worth it for a 5-minute call. |
| `distilbart-cnn-12-6` for summarization | One of the smallest summarization models that still produces decent English. The user can still upgrade to `gpt-4o-mini` for better summaries by setting an API key. |
| Keyword-based KPIs instead of ML scoring | Transparent and easy to debug — you can look at the keyword lists and see exactly why a score went up or down. No need for labeled training data. The architecture is set up so each KPI can be replaced with an ML model later without changing the rest of the code. |
| Flask for the backend | Simplest Python framework that handles a JSON API and also serves the frontend static files in the same process. Easy to deploy or put in a Docker container later. |

## 4. What happens when you upload a file (request lifecycle)

Step by step, here's what happens from clicking "Run analysis" to seeing the results:

```
1. User picks an audio file, selects context (sales/support/general), clicks "Run analysis".
2. Browser sends a multipart POST request to /api/analyze.
3. The backend:
   a. Checks the file extension and size (must be ≤50 MB).
   b. Saves the file to /backend/uploads/<job_id>.<ext>.
   c. Runs VoiceAnalyzer.analyze(path, context) — this is the 5-stage pipeline.
   d. Saves the result as JSON to /backend/results/<job_id>.json.
   e. Sends back the JSON.
4. Frontend hides the loading spinner and renders:
   - KPI cards (CSAT, Agent score, Sentiment)
   - Summary as bullet points
   - Sentiment trajectory chart
   - Context-specific KPI grid
   - Keyword chips
   - Turn-by-turn transcript with each turn's sentiment color-coded
5. User can copy the transcript, download the JSON result, or start a new analysis.
```

## 5. What can go wrong and how I handle it

I tried to fail gracefully wherever I could. Here's what happens in each error case:

| What goes wrong | What happens |
| --- | --- |
| User uploads a non-audio file | Backend returns a 400 error, UI shows a red error message |
| File is bigger than 50 MB | Werkzeug (the Flask underlying library) rejects it with a 413 error, UI shows the message |
| OpenAI API key is set but the request fails | Falls back automatically to the local distilbart summarizer |
| distilbart fails (e.g. out of memory) | Falls back to the basic extractive summary |
| Whisper produces an empty transcript (e.g. all silence) | Pipeline returns an empty transcript, the summary becomes empty, sentiment is neutral, KPIs reflect zero turns. No crash. |

## 6. Where this can be extended

If I (or someone else) wanted to take this further, here are the obvious next steps:

- **Real speaker diarization** — replace `_label_speakers` with `pyannote.audio` for calls with more than two speakers, or for cases where the simple alternation rule doesn't work well.
- **Emotion detection** — add a fine-tuned RoBERTa-emotion model on top of VADER to detect specific emotions (anger, joy, confusion, etc.) rather than just positive/negative.
- **PII redaction** — pre-process the transcript with `presidio` or some regex patterns to remove sensitive info like names, emails, credit card numbers, phone numbers. Important if this goes to production.
- **Better storage** — right now I'm just saving JSON files to a folder. Switching to SQLite or Postgres would be straightforward since the JSON shape is already pretty flat.
- **User accounts / auth** — adding Flask-Login would allow multiple users to use the system without seeing each other's calls.
