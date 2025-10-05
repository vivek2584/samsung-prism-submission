import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLI = PROJECT_ROOT / "summarize_video.py"


def have_ffmpeg():
	return subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode == 0


@pytest.mark.skipif(not have_ffmpeg(), reason="ffmpeg is required for tests")
def test_cli_with_segments_json(tmp_path: Path):
	# 1) Create a 8-second synthetic video with a color source and tone
	video_path = tmp_path / "input.mp4"
	cmd = [
		"ffmpeg","-y","-hide_banner","-loglevel","error",
		"-f","lavfi","-i","color=c=blue:s=640x360:d=8",
		"-f","lavfi","-i","sine=frequency=440:duration=8",
		"-c:v","libx264","-pix_fmt","yuv420p",
		"-c:a","aac","-shortest", str(video_path)
	]
	r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
	assert r.returncode == 0, r.stderr

	# 2) Provide segments JSON covering parts of the 8 seconds with distinct texts
	segments = [
		{"start": 0.2, "end": 1.5, "text": "intro summary"},
		{"start": 2.0, "end": 3.5, "text": "key point one important"},
		{"start": 5.0, "end": 6.5, "text": "another critical highlight"},
	]
	segments_path = tmp_path / "segments.json"
	segments_path.write_text(json.dumps(segments), encoding="utf-8")

	# 3) Run CLI targeting 3 seconds total
	out_dir = tmp_path / "out"
	cmd = [sys.executable, str(CLI), str(video_path), "--out", str(out_dir), "--target", "3", "--segments-json", str(segments_path)]
	r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
	assert r.returncode == 0, r.stderr

	# 4) Validate outputs exist and have non-zero size
	summary_mp4 = out_dir / f"{video_path.stem}_summary.mp4"
	srt_file = out_dir / f"{video_path.stem}_summary.srt"
	assert summary_mp4.exists() and summary_mp4.stat().st_size > 0
	assert srt_file.exists() and srt_file.stat().st_size > 0

	# 5) Optionally burn captions and check
	cmd = [sys.executable, str(CLI), str(video_path), "--out", str(out_dir), "--target", "3", "--segments-json", str(segments_path), "--burn"]
	r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
	assert r.returncode == 0, r.stderr
	burned = out_dir / f"{video_path.stem}_summary_captions.mp4"
	assert burned.exists() and burned.stat().st_size > 0
