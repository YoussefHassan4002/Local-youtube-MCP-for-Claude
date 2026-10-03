"""MCP server exposing tools that let Claude read and 'watch' a YouTube video from its URL."""

import functools
import logging
import os
import sys

import anyio

try:  # mcp >= 2.0 renamed FastMCP to MCPServer (same API)
    from mcp.server.mcpserver import MCPServer as FastMCP
    from mcp.server.mcpserver.exceptions import ToolError
    from mcp.server.mcpserver.utilities.types import Image
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP, Image
    from mcp.server.fastmcp.exceptions import ToolError

from . import transcript as tr
from .timeutil import fmt_time, resolve_range
from .video import CACHE_ROOT, extract_frames, get_info, video_id_from_url

logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
for noisy in ("httpx", "huggingface_hub", "faster_whisper"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
log = logging.getLogger("youtube_mcp")

MAX_TRANSCRIPT_CHARS = int(os.environ.get("YOUTUBE_MCP_MAX_TRANSCRIPT_CHARS", 80_000))  # ~1 hour of fast speech
MAX_FRAMES_HARD = int(os.environ.get("YOUTUBE_MCP_MAX_FRAMES", 40))
FRAME_WIDTH = int(os.environ.get("YOUTUBE_MCP_FRAME_WIDTH", 512))
FRAME_JPEG_QUALITY = 5  # ffmpeg -q:v scale, 2 (best) .. 31 (worst)
# Claude Desktop rejects tool results over 1 MB ("Tool result is too large"). Text plus base64-encoded
# images in one response must fit under this, which leaves headroom for JSON overhead.
MAX_RESPONSE_BYTES = int(os.environ.get("YOUTUBE_MCP_MAX_RESPONSE_BYTES", 900_000))
MAX_DESCRIPTION_CHARS = 5_000

TimeArg = str | float | None

mcp = FastMCP(
    "youtube",
    instructions=(
        "Tools for understanding YouTube videos from a URL. Times (start/end) accept '90', '1:30', "
        "'1:02:03' or '1h2m3s'; omit them for the whole video. Use watch_video for a combined view "
        "(transcript + frames), get_transcript for text only, get_frames for visuals only."
    ),
)


async def _run(fn, *args):
    """Run blocking work (downloads, ffmpeg, Whisper) off the event loop."""
    return await anyio.to_thread.run_sync(lambda: fn(*args))


def _tool(fn):
    """Register a tool whose failures reach Claude as readable messages instead of a generic error."""

    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        try:
            return await fn(*args, **kwargs)
        except ToolError:
            raise
        except Exception as exc:
            log.exception("%s failed", fn.__name__)
            msg = str(exc).removeprefix("ERROR: ").strip() or type(exc).__name__
            raise ToolError(msg) from exc

    return mcp.tool(structured_output=False)(wrapper)


# ---------------------------------------------------------------- helpers


def _plan_timestamps(start: float, end: float, interval: float, max_frames: int) -> tuple[list[float], str | None]:
    """Timestamps every `interval` seconds in [start, end); spread evenly if that exceeds max_frames."""
    interval = max(1.0, float(interval))
    max_frames = max(1, min(int(max_frames), MAX_FRAMES_HARD))
    wanted = max(1, int((end - start - 1e-6) // interval) + 1)
    if wanted <= max_frames:
        return [start + i * interval for i in range(wanted)], None
    step = (end - start) / max_frames
    note = (
        f"Frame count capped: an interval of {interval:g}s would need {wanted} frames, but the limit is "
        f"{max_frames}. Sampled every ~{step:.1f}s instead. Narrow start/end or raise "
        f"max_frames (hard limit {MAX_FRAMES_HARD}) for denser coverage."
    )
    return [start + i * step for i in range(max_frames)], note


def _load_frames(video_id: str, timestamps: list[float]) -> tuple[list[tuple[float, bytes]], list[str]]:
    """Extract frames. Returns ((t, jpeg), ...) and notes about any that failed."""
    results = extract_frames(video_id, timestamps, FRAME_WIDTH, FRAME_JPEG_QUALITY)
    failed = [t for t, p in results if p is None]
    notes = [f"Could not extract frames at: {', '.join(fmt_time(t) for t in failed)}."] if failed else []
    return [(t, p.read_bytes()) for t, p in results if p is not None], notes


def _fit_frames(frames: list[tuple[float, bytes]], text_bytes: int) -> tuple[list[tuple[float, bytes]], str | None]:
    """Keep frames, in order, while the text plus base64-encoded images stays under MAX_RESPONSE_BYTES."""
    used = text_bytes
    for i, (t, data) in enumerate(frames):
        used += (len(data) + 2) // 3 * 4 + 200  # base64 size, plus the frame's label and JSON wrapper
        if used > MAX_RESPONSE_BYTES:
            return frames[:i], (
                f"Response size limit ({MAX_RESPONSE_BYTES // 1000} KB; Claude Desktop rejects tool results over "
                f"1 MB) reached: dropped {len(frames) - i} frame(s) from {fmt_time(t)} onward. Request a "
                f"narrower range to see them."
            )
    return frames, None


def _range_label(start: float, end: float, duration: float | None) -> str:
    whole = start == 0 and duration is not None and abs(end - duration) < 1
    return f"{fmt_time(start)}–{fmt_time(end)}" + (" (whole video)" if whole else "")


def _transcript_segments(video_id: str, start: float, end: float, duration, language) -> tuple[list[dict], str]:
    t = tr.get_transcript(video_id, end, duration, language)
    return tr.segments_in_range(t, start, end), t.source


def _cap_lines(lines: list[tuple[float, str]], used: int = 0) -> tuple[list[tuple[float, str]], float | None]:
    """Keep transcript lines up to MAX_TRANSCRIPT_CHARS. Returns the kept lines and the time of the
    first line left out (None if everything fit). Always keeps at least one line."""
    for i, (t, line) in enumerate(lines):
        if i and used + len(line) + 1 > MAX_TRANSCRIPT_CHARS:
            return lines[:i], t
        used += len(line) + 1
    return lines, None


def _truncation_note(stopped_at: float, end: float, url: str) -> str:
    return (
        f"[TRUNCATED: transcript output limit ({MAX_TRANSCRIPT_CHARS:,} chars) reached at {fmt_time(stopped_at)}; "
        f"the requested range continues to {fmt_time(end)}. To read on, call get_transcript(url={url!r}, "
        f"start={fmt_time(stopped_at)!r}, end={fmt_time(end)!r}).]"
    )


# ---------------------------------------------------------------- tools


@_tool
async def get_video_info(url: str) -> str:
    """Get a YouTube video's title, channel, duration, upload date, description and chapters."""
    video_id = video_id_from_url(url)
    info = await _run(get_info, video_id)
    duration = info.get("duration")
    out = [
        f"Title: {info['title']}",
        f"Channel: {info['channel']}",
        f"Duration: {fmt_time(duration) if duration else 'unknown (live?)'}",
        f"Uploaded: {info['upload_date'] or 'unknown'}",
        f"Views: {info['view_count']:,}" if info.get("view_count") is not None else "Views: unknown",
        f"URL: {info['url']}",
    ]
    if info["chapters"]:
        out.append("\nChapters:")
        out += [f"  {fmt_time(c['start'] or 0)} {c['title']}" for c in info["chapters"]]
    else:
        out.append("\nChapters: none")
    desc = info["description"]
    if len(desc) > MAX_DESCRIPTION_CHARS:
        desc = desc[:MAX_DESCRIPTION_CHARS] + f"\n[TRUNCATED: description is {len(info['description']):,} chars]"
    out.append(f"\nDescription:\n{desc or '(empty)'}")
    return "\n".join(out)


@_tool
async def get_transcript(url: str, start: TimeArg = None, end: TimeArg = None, language: str | None = None) -> str:
    """Get the timestamped transcript of a YouTube video, optionally limited to a time range.

    Uses YouTube captions when available; otherwise transcribes the audio locally with Whisper
    (slower on first call, cached afterwards).

    Args:
        url: YouTube video URL (watch, youtu.be, shorts, embed, live) or bare video id.
        start: Range start, e.g. "90", "1:30", "1:02:03", "1h2m3s". Omit for the beginning.
        end: Range end, same formats. Omit for the end of the video ("watch until 12:30" -> end="12:30").
        language: Optional caption language code (e.g. "en", "es"). Default: English if available,
            otherwise the video's own language.
    """
    video_id = video_id_from_url(url)
    info = await _run(get_info, video_id)
    duration = info.get("duration")
    s, e = resolve_range(start, end, duration)
    segments, source = await _run(_transcript_segments, video_id, s, e, duration, language)
    lines = tr.format_lines(segments)

    header = f"Transcript of \"{info['title']}\" — {_range_label(s, e, duration)}\nSource: {source}\n"
    if not lines:
        return header + "\n(no speech found in this range)"
    kept, stopped_at = _cap_lines(lines, used=len(header))
    body = [line for _, line in kept]
    if stopped_at is not None:
        body.append(_truncation_note(stopped_at, e, url))
    return header + "\n" + "\n".join(body)


@_tool
async def get_frames(
    url: str,
    start: TimeArg = None,
    end: TimeArg = None,
    interval_seconds: float = 30,
    max_frames: int = 20,
) -> list:
    """Get still frames from a YouTube video as images, sampled every `interval_seconds` over a range.

    Downloads a low-res copy once (cached), then extracts frames with ffmpeg. If the interval would
    produce more than `max_frames` frames, frames are spread evenly across the range instead.

    Args:
        url: YouTube video URL or bare video id.
        start: Range start, e.g. "90", "1:30", "1:02:03", "1h2m3s". Omit for the beginning.
        end: Range end, same formats. Omit for the end of the video.
        interval_seconds: Seconds between frames (min 1). Default 30.
        max_frames: Maximum number of frames to return (default 20, hard limit 40).
    """
    video_id = video_id_from_url(url)
    info = await _run(get_info, video_id)
    duration = info.get("duration")
    s, e = resolve_range(start, end, duration)
    timestamps, cap_note = _plan_timestamps(s, e, interval_seconds, max_frames)
    frames, notes = await _run(_load_frames, video_id, timestamps)
    frames, size_note = _fit_frames(frames, text_bytes=2_000)  # the header is the only other text
    if size_note:
        notes.append(size_note)
    if cap_note:
        notes.insert(0, cap_note)

    content: list = [
        f"{len(frames)} frame(s) from \"{info['title']}\" — {_range_label(s, e, duration)}, "
        f"at {', '.join(fmt_time(t) for t, _ in frames)}."
        + "".join(f"\nNOTE: {n}" for n in notes)
    ]
    for t, data in frames:
        content.append(f"Frame at {fmt_time(t)}:")
        content.append(Image(data=data, format="jpeg"))
    return content


@_tool
async def watch_video(
    url: str,
    start: TimeArg = None,
    end: TimeArg = None,
    frame_interval: float = 30,
    max_frames: int = 20,
    language: str | None = None,
) -> list:
    """'Watch' a YouTube video (or part of it): frames interleaved with the transcript of what is said
    around each frame, plus title/chapters. Best single call for understanding a video.

    Args:
        url: YouTube video URL or bare video id.
        start: Range start, e.g. "90", "1:30", "1:02:03", "1h2m3s". Omit for the beginning.
        end: Range end, same formats. Omit for the end ("watch until 12:30" -> end="12:30").
        frame_interval: Seconds between frames (min 1). Default 30.
        max_frames: Maximum frames (default 20, hard limit 40); frames are spread evenly if exceeded.
        language: Optional caption language code.
    """
    video_id = video_id_from_url(url)
    info = await _run(get_info, video_id)
    duration = info.get("duration")
    s, e = resolve_range(start, end, duration)
    timestamps, cap_note = _plan_timestamps(s, e, frame_interval, max_frames)

    notes = [cap_note] if cap_note else []
    # Transcript and frames are independent; fetch them concurrently and tolerate either failing.
    results: dict = {}

    async def fetch(key, fn, *args):
        try:
            results[key] = await _run(fn, *args)
        except Exception as exc:
            log.exception("%s failed for %s", key, video_id)
            results[key] = exc

    async with anyio.create_task_group() as tg:
        tg.start_soon(fetch, "transcript", _transcript_segments, video_id, s, e, duration, language)
        tg.start_soon(fetch, "frames", _load_frames, video_id, timestamps)

    segments, source = [], "unavailable"
    if isinstance(results["transcript"], Exception):
        notes.append(f"Transcript unavailable: {results['transcript']}")
    else:
        segments, source = results["transcript"]
    frames: list[tuple[float, bytes]] = []
    if isinstance(results["frames"], Exception):
        notes.append(f"Frames unavailable: {results['frames']}")
    else:
        frames, frame_notes = results["frames"]
        notes += frame_notes

    # Cap the transcript first; frames get whatever room is left in the response.
    kept_lines, stopped_at = _cap_lines(tr.format_lines(segments))
    if stopped_at is not None:
        segments = [seg for seg in segments if seg["start"] < stopped_at]
    text_bytes = sum(len(line.encode()) + 1 for _, line in kept_lines) + 3_000  # + header, notes, chapters
    frames, size_note = _fit_frames(frames, text_bytes)
    if size_note:
        notes.append(size_note)

    chapters = [c for c in info["chapters"] if (c["end"] or e) > s and (c["start"] or 0) < e]
    header = [
        f"Watching \"{info['title']}\" by {info['channel']} — {_range_label(s, e, duration)}"
        f" (video length {fmt_time(duration) if duration else 'unknown'})",
        f"Transcript source: {source}. Frames: {len(frames)}.",
    ]
    if chapters:
        header.append("Chapters in range: " + "; ".join(f"{fmt_time(c['start'] or 0)} {c['title']}" for c in chapters))
    header += [f"NOTE: {n}" for n in notes]
    content: list = ["\n".join(header)]

    # Interleave: each frame followed by the speech from its timestamp up to the next frame.
    # Speech before the first frame (or all of it, if there are no frames) goes in the first block.
    boundaries = [t for t, _ in frames] or [s]
    si, note_added = 0, False
    for i, start_t in enumerate(boundaries):
        next_t = boundaries[i + 1] if i + 1 < len(boundaries) else float("inf")
        if frames:
            content.append(f"── {fmt_time(start_t)} ──")
            content.append(Image(data=frames[i][1], format="jpeg"))
        window = []
        while si < len(segments) and segments[si]["start"] < next_t:
            window.append(segments[si])
            si += 1
        if window:
            content.append("\n".join(line for _, line in tr.format_lines(window)))
        elif frames and segments and (stopped_at is None or next_t <= stopped_at):
            content.append("(no speech in this segment)")
        if stopped_at is not None and not note_added and next_t > stopped_at:
            # The transcript was cut inside this window; later frames have no text in this response.
            content.append(_truncation_note(stopped_at, e, url))
            note_added = True
    if not segments and not isinstance(results["transcript"], Exception):
        content.append("(no speech found in this range)")
    return content


def main() -> None:
    log.info("Starting YouTube MCP server (cache: %s)", CACHE_ROOT)
    mcp.run()


if __name__ == "__main__":
    main()
