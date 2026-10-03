# YouTube MCP for Claude

A local [MCP](https://modelcontextprotocol.io) server that lets Claude understand a YouTube video from just its URL. It reads the metadata and the timestamped transcript, and it looks at frames from the video.

Built with the official MCP Python SDK. With `mcp` v2 the high-level server class is `MCPServer`, which is what FastMCP was renamed to. The code also runs on `mcp` 1.x, where the class is still called `FastMCP`.

## Tools

| Tool | What it returns |
|---|---|
| `get_video_info(url)` | Title, channel, duration, upload date, views, description, chapters |
| `get_transcript(url, start=None, end=None, language=None)` | A timestamped transcript. It uses YouTube captions when they exist. Otherwise it downloads the audio and transcribes it locally with faster-whisper. |
| `get_frames(url, start=None, end=None, interval_seconds=30, max_frames=20)` | JPEG frames taken every N seconds from a low-res copy of the video |
| `watch_video(url, start=None, end=None, frame_interval=30, max_frames=20, language=None)` | Frames with the transcript interleaved: each frame is followed by what's said until the next frame. Also returns the title and the chapters in range. |

**Time ranges.** `start` and `end` accept `"90"`, `"1:30"`, `"1:02:03"`, `"1h2m3s"`, `"2m"` or `"45s"`. If you leave one out, the range runs to that edge of the video. For example, "watch until 12:30" is just `end="12:30"`.

**URLs.** These forms work: `youtube.com/watch?v=…`, `youtu.be/…`, `/shorts/…`, `/embed/…`, `/live/…`, or a bare 11-character video id.

### Limits and truncation

There's no limit on video length. The caps below apply to a single response, so one call can't flood the conversation. Longer videos just take more calls, and every truncated response tells Claude where to pick up. For example, the full transcript of a 1-hour talk comes back in one call.

Whenever something is cut, the response says so explicitly:

- **Transcript:** 80,000 characters per call. That's about an hour of fast speech (a 1-hour talk measured 66,883 characters, or roughly 17–20k tokens), and slower speakers fit more. When a range is longer, the output ends with a `[TRUNCATED: … call get_transcript(url=…, start="12:34", end="30:00")]` line that tells Claude exactly where to continue.
- **Frames:** `max_frames` defaults to 20, with a hard limit of 40. If `interval_seconds` would need more frames than that, the frames are spread evenly across the range instead, and a `NOTE` gives the effective interval.
- **Response size:** each response is kept under 900 KB in total, text plus images, because Claude Desktop rejects tool results over 1 MB. The transcript gets its space first and frames fill the rest. Frames are 512 px wide and usually 10–25 KB each, so 20–40 frames plus an hour of transcript fit comfortably. If frames ever don't fit, the later ones are dropped and a `NOTE` says which.
- **Description:** 5,000 characters.

If `watch_video` can't get the transcript or the frames, it still returns the other part, with a note explaining what failed.

### Caching

Everything is cached under `$TMPDIR/youtube-mcp-cache/<video_id>/`: the metadata, the captions, the Whisper transcripts, the downloaded audio and video, and the extracted frames. Repeat calls return almost instantly. The first frames call on a video downloads the whole video at 360p, which can take a minute or more for long videos (86 seconds for a 1-hour talk in testing). To clear the cache, delete that folder (`rm -rf "$TMPDIR/youtube-mcp-cache"`). macOS also clears it on its own over time.

Whisper transcribes 10 minutes at a time and stops once it has covered the end of your range. A request for the first 5 minutes of a 2-hour video without captions therefore only transcribes the first 10 minutes. Later requests continue from where the last one stopped.

## Setup

Requirements: macOS or Linux, [uv](https://docs.astral.sh/uv/), and a JavaScript runtime (Node.js, Deno or Bun). yt-dlp now needs the JS runtime to read YouTube's player. You do **not** need to install ffmpeg: if no system ffmpeg is found, the server uses the static binary bundled with `imageio-ffmpeg`.

```bash
# 1. Install uv (skip if `uv --version` works)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install a JS runtime if you have none (skip if `node --version` works)
brew install node        # or: brew install deno

# 3. Install the project's dependencies (uv fetches Python 3.12 if needed)
cd "/path/to/Local youtube MCP for Claude"
uv sync
```

The first time a video has no captions, faster-whisper downloads its model (`base`, about 145 MB) from Hugging Face. After that it runs offline.

### Add it to Claude Desktop

1. In Claude Desktop, open **Settings → Developer → Edit Config**. This opens `~/Library/Application Support/Claude/claude_desktop_config.json`.
2. Add the server. Use **absolute paths**, because Claude Desktop launches servers with a minimal `PATH`. Run `which uv` to find your uv path.

```json
{
  "mcpServers": {
    "youtube": {
      "command": "/Users/youssef/.local/bin/uv",
      "args": [
        "--directory",
        "/Users/youssef/Desktop/Web Projects/Personal Project/Local youtube MCP for Claude",
        "run",
        "youtube-mcp"
      ]
    }
  }
}
```

If the file already has an `mcpServers` object, add the `"youtube": {…}` entry inside it.

3. Fully quit Claude Desktop (⌘Q) and reopen it. The four tools should show up under the tools/connectors menu in the chat box.

Then ask things like:

- "Summarize https://youtu.be/jNQXAC9IVRw"
- "Watch https://www.youtube.com/watch?v=… until 12:30 and tell me what's on the slides"
- "What does he say between 1:02:00 and 1:10:00?"

### Claude Code

```bash
claude mcp add youtube -- uv --directory "/path/to/Local youtube MCP for Claude" run youtube-mcp
```

Claude Code limits each tool result to 25,000 tokens by default, and images count toward that. An hour of transcript fits. A `watch_video` call over a whole hour with many frames may not; if Claude Code complains, start it with a higher limit, for example `MAX_MCP_OUTPUT_TOKENS=50000 claude`.

## Configuration (optional)

You can set these as environment variables, or in an `"env": {…}` block in the Claude Desktop config.

| Variable | Default | Meaning |
|---|---|---|
| `YOUTUBE_MCP_CACHE_DIR` | `$TMPDIR/youtube-mcp-cache` | Cache location |
| `YOUTUBE_MCP_WHISPER_MODEL` | `base` | faster-whisper model: `tiny`, `base`, `small`, `medium`, `large-v3`, … Larger models are more accurate and slower. |
| `YOUTUBE_MCP_MAX_TRANSCRIPT_CHARS` | `80000` | Transcript characters per response |
| `YOUTUBE_MCP_MAX_FRAMES` | `40` | Hard upper limit for `max_frames` |
| `YOUTUBE_MCP_FRAME_WIDTH` | `512` | Frame width in pixels |
| `YOUTUBE_MCP_MAX_RESPONSE_BYTES` | `900000` | Total size of one response, text plus base64 images. Keep it under 1 MB for Claude Desktop. |
| `YOUTUBE_MCP_COOKIES_FROM_BROWSER` | unset | For example `chrome` or `firefox`. Uses that browser's YouTube cookies for age-restricted or members-only videos. |

## Troubleshooting

- **The tools don't appear in Claude Desktop.** Check `~/Library/Logs/Claude/mcp-server-youtube.log`. Make sure the `uv` path is absolute and that `uv sync` has been run.
- **"Sign in to confirm you're not a bot" or download errors.** Upgrade yt-dlp with `uv lock --upgrade-package yt-dlp && uv sync`. If that doesn't help, set `YOUTUBE_MCP_COOKIES_FROM_BROWSER`.
- **No JS runtime found.** Install Node or Deno. The server finds them in `PATH`, `/opt/homebrew/bin`, `/usr/local/bin` and `~/.deno/bin`.
- **Testing outside Claude.** Run the MCP Inspector:
  `npx @modelcontextprotocol/inspector uv --directory "/path/to/Local youtube MCP for Claude" run youtube-mcp`

## Project layout

```
src/youtube_mcp/
  server.py      # MCP tools, response caps and truncation notes
  video.py       # URL parsing, yt-dlp metadata and downloads, ffmpeg frame grabs, cache
  transcript.py  # YouTube captions, falling back to chunked faster-whisper
  timeutil.py    # "1:02:03" / "1h2m3s" parsing and formatting
```
