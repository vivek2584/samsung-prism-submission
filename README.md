# AI Video Summarizer (ffmpeg + Whisper)

A simple Python CLI that:
- extracts audio from a video with ffmpeg
- transcribes using OpenAI Whisper
- scores transcript segments and selects the most important snippets to fit a target duration
- cuts and concatenates clips
- generates SRT captions for the summary and optionally burns them into the video

## Requirements
- ffmpeg installed and on PATH (`ffmpeg -version`)
- Python 3.9+
- PyTorch (CPU or CUDA). Install matching build from PyTorch.

## Install
```bash
python -m venv .venv
. .venv/Scripts/activate  # on Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Install a compatible torch build, e.g. CPU:
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Usage
```bash
python summarize_video.py path/to/input.mp4 --target 60 --model base --burn
```
- `--target`: desired summary length in seconds (default 60)
- `--model`: Whisper model size: tiny, base, small, medium, large
- `--device`: set explicit device like `cpu` or `cuda` (auto if omitted)
- `--segments-json`: path to JSON segments `[ {"start": float, "end": float, "text": str }, ... ]` to bypass Whisper (useful for tests)
- `--burn`: burn captions into the final video

Outputs are written to the `out/` directory by default:
- `<name>_summary.mp4`
- `<name>_summary.srt`
- `<name>_summary_captions.mp4` (if `--burn`)

## Testing
Run tests with pytest. ffmpeg must be available in PATH.
```bash
pytest -q
```
The CLI test synthesizes a short color+tone video and provides a small segments JSON to avoid running Whisper during tests. It validates that summary MP4/SRT are produced and that optional burned captions render.

## Setup Guide (Windows)
- Install ffmpeg via winget/choco/scoop or manual zip; verify `ffmpeg -version`.
- Create and activate a virtualenv; install `requirements.txt` then `torch` (CPU or CUDA build).
- If you hit PATH issues for ffmpeg, add `C:\ffmpeg\bin` to your User PATH and reopen your shell.

4) Verify on GitHub that the tag exists and is accessible.

## Notes
- Cutting is done with re-encode for accurate boundaries; concatenation uses ffmpeg concat demuxer.
- TF-IDF scoring is a simple baseline. You can swap in better ranking methods later (e.g., embeddings, audio energy, or scene changes).
