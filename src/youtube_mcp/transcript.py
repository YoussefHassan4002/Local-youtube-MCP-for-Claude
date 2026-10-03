"""Timestamped transcripts: YouTube captions first, local Whisper transcription as a fallback."""

import json
import logging
import os
import subprocess
import threading
from dataclasses import dataclass

import numpy as np

from youtube_transcript_api import NoTranscriptFound, TranscriptsDisabled, YouTubeTranscriptApi

from .timeutil import fmt_time
from .video import cache_dir, download_audio, ffmpeg_path, lock_for

log = logging.getLogger(__name__)

WHISPER_MODEL = os.environ.get("YOUTUBE_MCP_WHISPER_MODEL", "base")
WHISPER_CHUNK_SECONDS = 600  # transcribe (and cache progress) 10 minutes at a time
SAMPLE_RATE = 16_000
_whisper = None
_whisper_lock = threading.Lock()


class NoCaptions(Exception):
    pass


@dataclass
class Transcript:
    segments: list[dict]  # each {"start": float, "end": float, "text": str}
    source: str  # human-readable description of where it came from
    language: str | None
    covered_until: float | None = None  # Whisper only: transcribed up to here (None = whole video)


def _pick_caption_track(video_id: str, language: str | None):
    tracks = YouTubeTranscriptApi().list(video_id)
    if language:
        return tracks.find_transcript([language])
    preferred = ["en", "en-US", "en-GB"]
    for finder in (tracks.find_manually_created_transcript, tracks.find_generated_transcript):
        try:
            return finder(preferred)
        except NoTranscriptFound:
            pass
    # No English: take the first manual track, else the first auto-generated one.
    all_tracks = list(tracks)
    return next((t for t in all_tracks if not t.is_generated), None) or all_tracks[0]


def _fetch_captions(video_id: str, language: str | None) -> Transcript:
    path = cache_dir(video_id) / f"captions_{language or 'auto'}.json"
    if path.exists():
        data = json.loads(path.read_text())
        if data.get("unavailable"):
            raise NoCaptions(f"no captions for language {language or 'any'}")
        return Transcript(**data)
    try:
        track = _pick_caption_track(video_id, language)
    except (NoTranscriptFound, TranscriptsDisabled, IndexError):
        # Remember that this video has no captions so we don't ask again.
        path.write_text(json.dumps({"unavailable": True}))
        raise NoCaptions(f"no captions for language {language or 'any'}")
    fetched = track.fetch()
    segments = [
        {"start": s.start, "end": s.start + s.duration, "text": s.text.replace("\n", " ").strip()}
        for s in fetched
        if s.text.strip()
    ]
    kind = "auto-generated" if track.is_generated else "manual"
    transcript = Transcript(segments, f"YouTube captions ({kind}, {track.language})", track.language_code)
    path.write_text(json.dumps(transcript.__dict__))
    return transcript


def _get_whisper():
    global _whisper
    with _whisper_lock:
        if _whisper is None:
            from faster_whisper import WhisperModel

            log.info("Loading faster-whisper model %r", WHISPER_MODEL)
            _whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
        return _whisper


def _decode_audio(audio_path, offset: float, length: float):
    """Decode a slice of audio to 16 kHz mono float32 with ffmpeg (what Whisper expects)."""
    cmd = [
        ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-nostdin",
        "-ss", f"{offset:.3f}", "-t", f"{length:.3f}", "-i", str(audio_path),
        "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-",
    ]  # fmt: skip
    proc = subprocess.run(cmd, capture_output=True, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg could not decode audio: {proc.stderr.decode(errors='replace')[:500]}")
    return np.frombuffer(proc.stdout, dtype=np.float32)


def _whisper_transcribe(video_id: str, until: float, duration: float | None, language: str | None) -> Transcript:
    """Transcribe locally in fixed-size chunks, stopping once `until` is covered. Progress is cached,
    so asking for the first 10 minutes of a 2-hour video doesn't transcribe all of it, and later
    calls resume where the previous one stopped."""
    path = cache_dir(video_id) / f"whisper_{WHISPER_MODEL}_{language or 'auto'}.json"
    with lock_for(f"{video_id}:whisper"):
        t = Transcript(**json.loads(path.read_text())) if path.exists() else None
        if t and (t.covered_until is None or t.covered_until >= until):
            return t
        if t is None:
            source = f"local Whisper transcription (faster-whisper '{WHISPER_MODEL}'; no YouTube captions available)"
            t = Transcript([], source, language, covered_until=0.0)

        audio = download_audio(video_id)
        model = _get_whisper()
        while t.covered_until is not None and t.covered_until < until:
            offset = t.covered_until
            log.info("Whisper transcribing %s %.0fs-%.0fs", video_id, offset, offset + WHISPER_CHUNK_SECONDS)
            samples = _decode_audio(audio, offset, WHISPER_CHUNK_SECONDS)
            chunk_len = len(samples) / SAMPLE_RATE
            if chunk_len > 0.5:
                results, info = model.transcribe(samples, language=t.language, vad_filter=True)
                for seg in results:
                    if seg.text.strip():
                        t.segments.append({"start": offset + seg.start, "end": offset + seg.end, "text": seg.text.strip()})
                t.language = t.language or info.language  # keep the language detected in the first chunk
            reached_end = chunk_len < WHISPER_CHUNK_SECONDS - 1 or (
                duration is not None and offset + WHISPER_CHUNK_SECONDS >= duration
            )
            t.covered_until = None if reached_end else offset + WHISPER_CHUNK_SECONDS
            path.write_text(json.dumps(t.__dict__))
        return t


def get_transcript(video_id: str, until: float, duration: float | None, language: str | None = None) -> Transcript:
    try:
        return _fetch_captions(video_id, language)
    except Exception as exc:  # no captions, or the transcript API is blocked / failing
        log.info("Captions unavailable for %s (%s: %s); falling back to Whisper", video_id, type(exc).__name__, exc)
    return _whisper_transcribe(video_id, until, duration, language)


def segments_in_range(transcript: Transcript, start: float, end: float) -> list[dict]:
    return [s for s in transcript.segments if s["end"] > start and s["start"] < end]


def format_lines(segments: list[dict], group_seconds: float = 10.0) -> list[tuple[float, str]]:
    """Merge short caption snippets into ~10-second lines: (start_time, "[m:ss] text")."""
    lines: list[tuple[float, str]] = []
    cur_start, cur_text = None, []
    for seg in segments:
        if cur_start is not None and seg["start"] - cur_start >= group_seconds:
            lines.append((cur_start, f"[{fmt_time(cur_start)}] {' '.join(cur_text)}"))
            cur_start, cur_text = None, []
        if cur_start is None:
            cur_start = seg["start"]
        cur_text.append(seg["text"])
    if cur_start is not None:
        lines.append((cur_start, f"[{fmt_time(cur_start)}] {' '.join(cur_text)}"))
    return lines
