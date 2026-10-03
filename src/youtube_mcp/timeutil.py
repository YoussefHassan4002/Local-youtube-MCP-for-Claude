"""Parsing and formatting of video timestamps."""

import re

_HMS_RE = re.compile(r"(?:(\d+(?:\.\d+)?)h)?(?:(\d+(?:\.\d+)?)m)?(?:(\d+(?:\.\d+)?)s?)?")


def parse_time(value: str | int | float | None) -> float | None:
    """Parse "90", "1:30", "1:02:03", "1h2m3s", "2m", "45s" into seconds.

    None or an empty string means "not specified".
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if value < 0:
            raise ValueError(f"Time must not be negative: {value}")
        return float(value)

    text = value.strip().lower().replace(" ", "")
    if not text:
        return None

    if ":" in text:
        parts = text.split(":")
        if len(parts) > 3 or any(not re.fullmatch(r"\d+(?:\.\d+)?", p) for p in parts):
            raise ValueError(f"Could not parse time {value!r}; use e.g. '90', '1:30', '1:02:03' or '1h2m3s'")
        seconds = 0.0
        for part in parts:
            seconds = seconds * 60 + float(part)
        return seconds

    match = _HMS_RE.fullmatch(text)
    if not match or not any(match.groups()):
        raise ValueError(f"Could not parse time {value!r}; use e.g. '90', '1:30', '1:02:03' or '1h2m3s'")
    h, m, s = (float(g) if g else 0.0 for g in match.groups())
    return h * 3600 + m * 60 + s


def fmt_time(seconds: float) -> str:
    """Format seconds as M:SS or H:MM:SS."""
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def resolve_range(start, end, duration: float | None) -> tuple[float, float]:
    """Turn user-supplied start/end into a concrete, validated (start, end) range."""
    s = parse_time(start) or 0.0
    e = parse_time(end)
    if duration:
        if s >= duration:
            raise ValueError(f"start ({fmt_time(s)}) is past the end of the video ({fmt_time(duration)})")
        e = duration if e is None else min(e, duration)
    elif e is None:
        raise ValueError("Video duration is unknown (live stream?); please pass an explicit end time")
    if e <= s:
        raise ValueError(f"end ({fmt_time(e)}) must be after start ({fmt_time(s)})")
    return s, e
