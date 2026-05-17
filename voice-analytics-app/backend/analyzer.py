"""
analyzer.py
===========
Core analysis pipeline for the Voice Analytics application.

Pipeline:
  1. Transcribe audio → text + per-segment timestamps     (faster-whisper)
  2. Speaker-role heuristic (Agent / Customer)            (rule-based)
  3. Summarisation                                        (transformers OR OpenAI)
  4. Sentiment — overall + segment level                  (VADER)
  5. KPI extraction tailored to context (sales/support)   (rule-based + lexicon)

All heavy models are loaded lazily so the server boots quickly.
"""

from __future__ import annotations

import logging
import re
import os
from collections import Counter
from statistics import mean
from typing import Any

log = logging.getLogger("voice-analytics.analyzer")


# ------------------------------------------------------------------ #
#  Lexicons                                                          #
# ------------------------------------------------------------------ #
POSITIVE_CUSTOMER_CUES = {
    "thank you", "thanks", "great", "perfect", "excellent", "amazing",
    "appreciate", "wonderful", "love it", "happy", "satisfied",
    "that works", "sounds good", "exactly", "awesome", "brilliant",
}

NEGATIVE_CUSTOMER_CUES = {
    "frustrated", "annoyed", "angry", "useless", "terrible", "awful",
    "worst", "disappointed", "complaint", "issue", "problem", "broken",
    "not working", "doesn't work", "horrible", "ridiculous", "waste",
    "cancel", "refund", "speak to manager", "escalate",
}

AGENT_GOOD_BEHAVIOURS = {
    "let me check", "i can help", "i understand", "i apologise", "i apologize",
    "right away", "absolutely", "happy to", "let me look into",
    "thank you for your patience", "i'll take care of",
    "is there anything else", "to confirm", "just to recap",
}

AGENT_POOR_BEHAVIOURS = {
    "i don't know", "not my problem", "you'll have to", "calm down",
    "that's not possible", "policy is policy", "as i already said",
}

SALES_INTENT_CUES = {
    "interested in", "tell me more", "send me a quote", "send me the proposal",
    "ready to buy", "let's move forward", "sign up", "subscribe",
    "what's the price", "pricing", "discount", "demo",
}

SUPPORT_RESOLUTION_CUES = {
    "issue resolved", "that fixed it", "working now", "all good",
    "ticket closed", "case closed", "appreciate the help",
}

FILLER_WORDS = {"um", "uh", "like", "you know", "basically", "literally", "kind of", "sort of"}


# ------------------------------------------------------------------ #
#  Analyzer                                                          #
# ------------------------------------------------------------------ #
class VoiceAnalyzer:
    """Lazy-loading wrapper around the full pipeline."""

    def __init__(self):
        self._whisper = None
        self._sentiment = None
        self._summarizer = None
        self._use_openai = bool(os.environ.get("OPENAI_API_KEY"))

    # -------------------------- model loaders ---------------------- #
    def _get_whisper(self):
        if self._whisper is None:
            from faster_whisper import WhisperModel
            # `base` is a good speed/accuracy trade-off; CPU int8 for portability.
            log.info("Loading faster-whisper model (base, int8)…")
            self._whisper = WhisperModel("base", device="cpu", compute_type="int8")
        return self._whisper

    def _get_sentiment(self):
        if self._sentiment is None:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._sentiment = SentimentIntensityAnalyzer()
        return self._sentiment

    def _get_summarizer(self):
        """Local summarizer using HuggingFace transformers (lazy)."""
        if self._summarizer is None and not self._use_openai:
            from transformers import pipeline
            log.info("Loading local summarisation pipeline (distilbart-cnn-12-6)…")
            self._summarizer = pipeline(
                "summarization",
                model="sshleifer/distilbart-cnn-12-6",
            )
        return self._summarizer

    # ============================================================== #
    #  PUBLIC ENTRY                                                  #
    # ============================================================== #
    def analyze(self, audio_path: str, context: str = "general") -> dict[str, Any]:
        log.info("[1/5] Transcribing %s", audio_path)
        transcript, segments, duration = self._transcribe(audio_path)

        log.info("[2/5] Labelling speaker roles")
        labelled = self._label_speakers(segments)

        log.info("[3/5] Generating summary")
        summary = self._summarise(transcript)

        log.info("[4/5] Running sentiment analysis")
        sentiment = self._sentiment_overall(transcript, labelled)

        log.info("[5/5] Extracting KPIs (context=%s)", context)
        kpis = self._kpis(transcript, labelled, sentiment, duration, context)

        return {
            "duration_seconds": round(duration, 2),
            "word_count": len(transcript.split()),
            "transcript": transcript,
            "segments": labelled,
            "summary": summary,
            "sentiment": sentiment,
            "kpis": kpis,
        }

    # ============================================================== #
    #  STEP 1 — transcription                                         #
    # ============================================================== #
    def _transcribe(self, audio_path: str):
        model = self._get_whisper()
        segments_iter, info = model.transcribe(audio_path, beam_size=5, vad_filter=True)
        segments = []
        full_text_parts = []
        for seg in segments_iter:
            text = seg.text.strip()
            if not text:
                continue
            segments.append({
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": text,
            })
            full_text_parts.append(text)
        transcript = " ".join(full_text_parts)
        duration = float(getattr(info, "duration", 0.0) or 0.0)
        return transcript, segments, duration

    # ============================================================== #
    #  STEP 2 — speaker-role heuristic                                #
    # ============================================================== #
    def _label_speakers(self, segments: list[dict]) -> list[dict]:
        """
        Lightweight rule-based role labelling. We do NOT do true diarization
        (which needs pyannote + HF token). Instead we alternate turns and use
        linguistic cues — the first speaker who greets / introduces themselves
        is treated as the Agent.
        """
        if not segments:
            return []

        # Heuristic 1: first turn that contains a self-introduction or greeting → Agent
        agent_first = True
        first_text = segments[0]["text"].lower()
        introductory = ("how can i help", "how may i", "thank you for calling",
                        "this is", "speaking", "welcome to")
        if not any(p in first_text for p in introductory):
            # If the opening sounds like a customer (problem statement), flip.
            problem_openers = ("hi i", "hello i", "i have", "i'm calling", "i need",
                               "i want", "my", "your service", "your product")
            if any(first_text.startswith(p) for p in problem_openers):
                agent_first = False

        labelled = []
        for i, seg in enumerate(segments):
            # Alternate turns, but merge consecutive sentences of similar style
            role_index = (i + (0 if agent_first else 1)) % 2
            role = "Agent" if role_index == 0 else "Customer"
            sent = self._get_sentiment().polarity_scores(seg["text"])
            labelled.append({**seg, "speaker": role, "sentiment_score": sent["compound"]})
        return labelled

    # ============================================================== #
    #  STEP 3 — summarisation                                         #
    # ============================================================== #
    def _summarise(self, transcript: str) -> dict[str, Any]:
        if not transcript.strip():
            return {"text": "", "bullets": []}

        # ----- Option A: OpenAI (if API key provided) ----- #
        if self._use_openai:
            try:
                from openai import OpenAI
                client = OpenAI()
                prompt = (
                    "Summarise this call recording in 4-6 bullet points covering: "
                    "main topic, customer's issue/intent, what the agent did, "
                    "any commitments made, and outcome. Be specific.\n\n"
                    f"TRANSCRIPT:\n{transcript[:12000]}"
                )
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                )
                text = resp.choices[0].message.content.strip()
                bullets = [b.lstrip("-• ").strip() for b in text.split("\n") if b.strip()]
                return {"text": text, "bullets": bullets, "engine": "openai-gpt4o-mini"}
            except Exception as e:
                log.warning("OpenAI summarisation failed (%s) — falling back to local.", e)

        # ----- Option B: Local transformer ----- #
        try:
            summariser = self._get_summarizer()
            # The model has a 1024-token limit; chunk if necessary.
            chunks = self._chunk_text(transcript, max_words=700)
            summaries = []
            for chunk in chunks:
                out = summariser(
                    chunk,
                    max_length=130,
                    min_length=40,
                    do_sample=False,
                )
                summaries.append(out[0]["summary_text"].strip())
            joined = " ".join(summaries)
            bullets = self._to_bullets(joined)
            return {"text": joined, "bullets": bullets, "engine": "distilbart-cnn"}
        except Exception as e:
            log.warning("Local summariser failed (%s) — using extractive fallback.", e)
            return self._extractive_summary(transcript)

    @staticmethod
    def _chunk_text(text: str, max_words: int = 700) -> list[str]:
        words = text.split()
        return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)] or [text]

    @staticmethod
    def _to_bullets(text: str) -> list[str]:
        # Split summary text into sentence bullets
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [p for p in parts if len(p.split()) > 3]

    def _extractive_summary(self, transcript: str) -> dict[str, Any]:
        """Last-resort summary: pick the most representative sentences."""
        sentences = re.split(r"(?<=[.!?])\s+", transcript)
        scored = []
        words = re.findall(r"\w+", transcript.lower())
        freq = Counter(w for w in words if len(w) > 3)
        for s in sentences:
            ws = re.findall(r"\w+", s.lower())
            if not ws:
                continue
            score = sum(freq.get(w, 0) for w in ws) / (len(ws) ** 0.5)
            scored.append((score, s.strip()))
        scored.sort(reverse=True)
        top = [s for _, s in scored[:5]]
        return {"text": " ".join(top), "bullets": top, "engine": "extractive-fallback"}

    # ============================================================== #
    #  STEP 4 — sentiment                                             #
    # ============================================================== #
    def _sentiment_overall(self, transcript: str, segments: list[dict]) -> dict[str, Any]:
        vader = self._get_sentiment()
        overall = vader.polarity_scores(transcript)

        # Per-speaker average compound score
        cust_scores = [s["sentiment_score"] for s in segments if s["speaker"] == "Customer"]
        agent_scores = [s["sentiment_score"] for s in segments if s["speaker"] == "Agent"]

        def _label(compound: float) -> str:
            if compound >= 0.25:
                return "Positive"
            if compound <= -0.25:
                return "Negative"
            return "Neutral"

        # Trajectory: average sentiment across thirds of the call
        trajectory = []
        if segments:
            n = len(segments)
            third = max(1, n // 3)
            buckets = [segments[:third], segments[third:2 * third], segments[2 * third:]]
            labels = ["Start", "Middle", "End"]
            for label, bucket in zip(labels, buckets):
                if bucket:
                    avg = mean(s["sentiment_score"] for s in bucket)
                    trajectory.append({"phase": label, "score": round(avg, 3), "label": _label(avg)})

        return {
            "overall": {
                "compound": round(overall["compound"], 3),
                "positive": round(overall["pos"], 3),
                "neutral": round(overall["neu"], 3),
                "negative": round(overall["neg"], 3),
                "label": _label(overall["compound"]),
            },
            "customer": {
                "average": round(mean(cust_scores), 3) if cust_scores else 0.0,
                "label": _label(mean(cust_scores)) if cust_scores else "Unknown",
                "sample_count": len(cust_scores),
            },
            "agent": {
                "average": round(mean(agent_scores), 3) if agent_scores else 0.0,
                "label": _label(mean(agent_scores)) if agent_scores else "Unknown",
                "sample_count": len(agent_scores),
            },
            "trajectory": trajectory,
        }

    # ============================================================== #
    #  STEP 5 — KPIs                                                  #
    # ============================================================== #
    def _kpis(self, transcript: str, segments: list[dict], sentiment: dict,
              duration: float, context: str) -> dict[str, Any]:
        text_lower = transcript.lower()

        # --- Customer Satisfaction Indicators ---
        positive_hits = sum(text_lower.count(cue) for cue in POSITIVE_CUSTOMER_CUES)
        negative_hits = sum(text_lower.count(cue) for cue in NEGATIVE_CUSTOMER_CUES)
        cust_compound = sentiment["customer"]["average"]

        # CSAT estimate (0–100). Combines lexicon hits + customer sentiment.
        raw = 50 + (cust_compound * 50) + (positive_hits - negative_hits) * 3
        csat = max(0, min(100, round(raw)))

        # --- Agent Performance ---
        agent_text = " ".join(s["text"].lower() for s in segments if s["speaker"] == "Agent")
        cust_text = " ".join(s["text"].lower() for s in segments if s["speaker"] == "Customer")
        good_behaviour = sum(agent_text.count(cue) for cue in AGENT_GOOD_BEHAVIOURS)
        poor_behaviour = sum(agent_text.count(cue) for cue in AGENT_POOR_BEHAVIOURS)
        filler_count = sum(agent_text.count(w) for w in FILLER_WORDS)

        agent_word_count = max(1, len(agent_text.split()))
        cust_word_count = max(1, len(cust_text.split()))
        talk_listen_ratio = round(agent_word_count / (agent_word_count + cust_word_count), 2)

        # Agent score (0–100)
        agent_raw = (
            50
            + good_behaviour * 5
            - poor_behaviour * 10
            - filler_count * 0.5
            + sentiment["agent"]["average"] * 20
        )
        # Penalise extreme talk dominance (ideal ≈ 0.4–0.6 for the agent)
        if talk_listen_ratio > 0.7 or talk_listen_ratio < 0.3:
            agent_raw -= 8
        agent_score = max(0, min(100, round(agent_raw)))

        # --- Context-specific KPIs ---
        context_kpis = {}
        if context == "sales":
            interest_hits = sum(text_lower.count(c) for c in SALES_INTENT_CUES)
            objections = sum(text_lower.count(c) for c in {"too expensive", "not interested", "maybe later", "think about it"})
            commitment = any(p in text_lower for p in ("send me", "sign up", "let's move forward", "demo", "follow up"))
            context_kpis = {
                "buying_intent_signals": interest_hits,
                "objection_signals": objections,
                "commitment_secured": commitment,
                "lead_quality": "Hot" if interest_hits >= 3 and objections <= 1
                                else "Warm" if interest_hits >= 1
                                else "Cold",
            }
        elif context == "support":
            resolution_hits = sum(text_lower.count(c) for c in SUPPORT_RESOLUTION_CUES)
            escalation = any(c in text_lower for c in ("escalate", "speak to manager", "supervisor"))
            issue_keywords = self._top_keywords(cust_text, 5)
            context_kpis = {
                "resolution_signals": resolution_hits,
                "escalation_requested": escalation,
                "likely_resolved": resolution_hits > 0 and not escalation,
                "top_customer_concerns": issue_keywords,
            }

        # --- Topics / keywords ---
        keywords = self._top_keywords(transcript, 10)

        return {
            "customer_satisfaction": {
                "score": csat,
                "label": "High" if csat >= 70 else "Medium" if csat >= 40 else "Low",
                "positive_cue_count": positive_hits,
                "negative_cue_count": negative_hits,
            },
            "agent_performance": {
                "score": agent_score,
                "label": "Strong" if agent_score >= 70 else "Adequate" if agent_score >= 40 else "Needs coaching",
                "good_behaviours_used": good_behaviour,
                "poor_behaviours_used": poor_behaviour,
                "filler_word_count": filler_count,
                "talk_ratio": talk_listen_ratio,
            },
            "conversation_metrics": {
                "duration_seconds": round(duration, 2),
                "duration_minutes": round(duration / 60, 2) if duration else 0,
                "turn_count": len(segments),
                "agent_word_count": agent_word_count,
                "customer_word_count": cust_word_count,
                "words_per_minute": round((agent_word_count + cust_word_count) / (duration / 60), 1) if duration > 1 else 0,
            },
            "top_keywords": keywords,
            "context_specific": context_kpis,
        }

    # ============================================================== #
    #  Utility — keyword extraction                                   #
    # ============================================================== #
    @staticmethod
    def _top_keywords(text: str, n: int = 10) -> list[str]:
        stop = {
            "the", "and", "you", "your", "that", "this", "have", "for", "with",
            "are", "was", "were", "but", "not", "they", "them", "their", "from",
            "what", "when", "will", "would", "could", "should", "about", "just",
            "like", "yeah", "okay", "well", "know", "think", "going", "really",
            "right", "want", "need", "get", "got", "can", "all", "any", "out",
            "one", "two", "now", "say", "said", "see", "tell", "told", "let",
            "make", "made", "back", "good", "way", "yes", "yep", "okay", "ok",
            "thank", "thanks", "please", "i'm", "i'll", "i've", "don't", "it's",
            "you're", "we're", "we've", "we'll", "that's", "there", "here",
        }
        words = re.findall(r"[a-zA-Z']{4,}", text.lower())
        words = [w for w in words if w not in stop]
        freq = Counter(words)
        return [w for w, _ in freq.most_common(n)]
