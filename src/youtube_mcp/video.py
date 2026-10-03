"""YouTube metadata, downloads (via yt-dlp) and frame extraction (via ffmpeg), all cached on disk."""

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import yt_dlp

log = logging.getLogger(__name__)

CACHE_ROOT = Path(os.environ.get("YOUTUBE_MCP_CACHE_DIR") or Path(tempfile.gettempdir()) / "youtube-mcp-cache")

_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def lock_for(key: str) -> threading.Lock:
    """One lock per cache key, so concurrent calls don't download the same file twice."""
    with _locks_guard:
        return _locks.setdefault(key, threading.Lock())


def video_id_from_url(url: str) -> str:
    """Extract the 11-character video id from any common YouTube URL form (or a bare id)."""
    url = url.strip()
    if _VIDEO_ID_RE.match(url):
        return url
    parsed = urlparse(url if "://" in url else "https://" + url)
    host = (parsed.hostname or "").lower().removeprefix("www.").removeprefix("m.")
    candidate = None
    if host == "youtu.be":
        candidate = parsed.path.lstrip("/").split("/")[0]
    elif host.endswith("youtube.com") or host.endswith("youtube-nocookie.com"):
        if parsed.path == "/watch":
            candidate = parse_qs(parsed.query).get("v", [None])[0]
        else:
            parts = parsed.path.strip("/").split("/")
            if len(parts) >= 2 and parts[0] in ("shorts", "embed", "live", "v", "e"):
                candidate = parts[1]
    if candidate and _VIDEO_ID_RE.match(candidate):
        return candidate
    raise ValueError(f"Not a recognizable YouTube video URL: {url!r}")


def cache_dir(video_id: str) -> Path:
    path = CACHE_ROOT / video_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def ffmpeg_path() -> str:
    """Prefer a system ffmpeg; fall back to the static binary shipped with imageio-ffmpeg."""
    found = shutil.which("ffmpeg") or next(
        (p for p in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg") if os.path.exists(p)), None
    )
    if found:
        return found
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _js_runtimes() -> dict:
    """yt-dlp needs a JS runtime to solve YouTube's player challenges.

    Claude Desktop launches servers with a minimal PATH, so also look in the usual install locations.
    """
    home = Path.home()
    candidates = {
        "deno": [home / ".deno/bin/deno", Path("/opt/homebrew/bin/deno"), Path("/usr/local/bin/deno")],
        "node": [Path("/opt/homebrew/bin/node"), Path("/usr/local/bin/node")],
        "bun": [home / ".bun/bin/bun", Path("/opt/homebrew/bin/bun")],
    }
    runtimes = {}
    for name, paths in candidates.items():
        found = shutil.which(name) or next((str(p) for p in paths if p.exists()), None)
        if found:
            runtimes[name] = {"path": found}
    return runtimes or {"deno": {}}


class _StderrLogger:
    """Route yt-dlp output to our logger; stdout is reserved for the MCP stdio protocol."""

    def debug(self, msg):
        pass

    def info(self, msg):
        pass

    def warning(self, msg):
        log.warning("yt-dlp: %s", msg)

    def error(self, msg):
        log.error("yt-dlp: %s", msg)


def _ydl_opts(**extra) -> dict:
    opts = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "noplaylist": True,
        "logger": _StderrLogger(),
        "js_runtimes": _js_runtimes(),
        "ffmpeg_location": ffmpeg_path(),
    }
    if browser := os.environ.get("YOUTUBE_MCP_COOKIES_FROM_BROWSER"):
        opts["cookiesfrombrowser"] = (browser,)
    opts.update(extra)
    return opts


def get_info(video_id: str) -> dict:
    """Video metadata (title, channel, duration, description, chapters), cached as JSON."""
    path = cache_dir(video_id) / "info.json"
    with lock_for(f"{video_id}:info"):
        if path.exists():
            return json.loads(path.read_text())
        with yt_dlp.YoutubeDL(_ydl_opts(skip_download=True)) as ydl:
            raw = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
        info = {
            "id": video_id,
            "title": raw.get("title"),
            "channel": raw.get("channel") or raw.get("uploader"),
            "channel_url": raw.get("channel_url"),
            "duration": raw.get("duration"),
            "upload_date": raw.get("upload_date"),
            "view_count": raw.get("view_count"),
            "is_live": raw.get("is_live"),
            "description": raw.get("description") or "",
            "chapters": [
                {"start": c.get("start_time"), "end": c.get("end_time"), "title": c.get("title")}
                for c in raw.get("chapters") or []
            ],
            "url": raw.get("webpage_url") or f"https://www.youtube.com/watch?v={video_id}",
        }
        path.write_text(json.dumps(info))
        return info


def _find_download(directory: Path, stem: str) -> Path | None:
    for p in directory.glob(f"{stem}.*"):
        if p.suffix not in (".part", ".ytdl", ".json") and not p.name.endswith(".part"):
            return p
    return None


def _download(video_id: str, stem: str, fmt: str) -> Path:
    directory = cache_dir(video_id)
    with lock_for(f"{video_id}:{stem}"):
        if existing := _find_download(directory, stem):
            return existing
        log.info("Downloading %s for %s", stem, video_id)
        opts = _ydl_opts(format=fmt, outtmpl=str(directory / f"{stem}.%(ext)s"), socket_timeout=30)
        for attempt in range(3):  # yt-dlp resumes the .part file, so retries are cheap
            try:
                with yt_dlp.YoutubeDL(opts) as ydl:
                    ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
                break
            except yt_dlp.utils.DownloadError as exc:
                if attempt == 2 or "timed out" not in str(exc).lower() and "connection" not in str(exc).lower():
                    raise
                log.warning("Download of %s %s failed (%s); retrying", video_id, stem, exc)
        if found := _find_download(directory, stem):
            return found
        raise RuntimeError(f"yt-dlp finished but no {stem} file was produced for {video_id}")


def download_video(video_id: str) -> Path:
    """Low-res (<=360p) video-only stream, good enough for frame grabs. Prefers H.264 for fast seeking."""
    return _download(
        video_id,
        "video",
        "bv*[height<=360][vcodec^=avc1]/bv*[height<=360]/b[height<=360]/wv*/w",
    )


def download_audio(video_id: str) -> Path:
    """Smallest reasonable audio stream; faster-whisper decodes it directly, no conversion needed."""
    return _download(video_id, "audio", "ba[abr<=96]/ba/w")


def extract_frames(video_id: str, timestamps: list[float], width: int, quality: int) -> list[tuple[float, Path | None]]:
    """Grab one JPEG per timestamp (cached). Returns (timestamp, path or None if extraction failed)."""
    video = download_video(video_id)
    frames_dir = cache_dir(video_id) / "frames"
    frames_dir.mkdir(exist_ok=True)
    ffmpeg = ffmpeg_path()

    def grab(t: float) -> tuple[float, Path | None]:
        out = frames_dir / f"w{width}_q{quality}_{t:.2f}.jpg"
        if out.exists() and out.stat().st_size > 0:
            return t, out
        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
            "-ss", f"{t:.3f}", "-i", str(video),
            "-frames:v", "1", "-vf", f"scale={width}:-2", "-q:v", str(quality),
            "-y", str(out),
        ]  # fmt: skip
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if proc.returncode != 0 or not out.exists() or out.stat().st_size == 0:
            log.warning("ffmpeg failed at %.2fs: %s", t, proc.stderr.strip()[:500])
            out.unlink(missing_ok=True)
            return t, None
        return t, out

    with ThreadPoolExecutor(max_workers=4) as pool:
        return list(pool.map(grab, timestamps))
