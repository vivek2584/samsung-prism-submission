import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

try:
	import whisper
except ImportError as e:
	whisper = None


@dataclass
class Segment:
	start: float
	end: float
	text: str
	index: int


def run_ffmpeg(args: List[str]) -> None:
	cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"] + args
	result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
	if result.returncode != 0:
		raise RuntimeError(f"ffmpeg failed: {' '.join(cmd)}\n{result.stderr}")


def extract_audio(input_video: Path, out_wav: Path) -> None:
	# Mono 16kHz WAV for Whisper
	run_ffmpeg(["-i", str(input_video), "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", str(out_wav)])


def transcribe_audio(audio_path: Path, model_name: str, device: str) -> List[Segment]:
	if whisper is None:
		raise ImportError("openai-whisper is not installed. Install it or provide --segments-json for testing.")
	model = whisper.load_model(model_name, device=device if device else None)
	# Use default language detection and timestamps
	result = model.transcribe(str(audio_path), verbose=False)
	segments: List[Segment] = []
	for idx, seg in enumerate(result.get("segments", [])):
		start = float(seg.get("start", 0.0))
		end = float(seg.get("end", 0.0))
		text = (seg.get("text") or "").strip()
		if end > start and text:
			segments.append(Segment(start=start, end=end, text=text, index=idx))
	return segments


def load_segments_from_json(json_path: Path) -> List[Segment]:
	data = json.loads(Path(json_path).read_text(encoding="utf-8"))
	segments: List[Segment] = []
	for idx, item in enumerate(data):
		start = float(item["start"]) if isinstance(item, dict) else float(item[0])
		end = float(item["end"]) if isinstance(item, dict) else float(item[1])
		text = (item.get("text") if isinstance(item, dict) else (item[2] if len(item) > 2 else "")).strip()
		if end > start:
			segments.append(Segment(start=start, end=end, text=text, index=idx))
	return segments


def merge_adjacent_ranges(ranges: List[Tuple[float, float]], max_gap: float = 0.5) -> List[Tuple[float, float]]:
	if not ranges:
		return []
	ranges = sorted(ranges)
	merged: List[Tuple[float, float]] = []
	cur_s, cur_e = ranges[0]
	for s, e in ranges[1:]:
		if s - cur_e <= max_gap:
			cur_e = max(cur_e, e)
		else:
			merged.append((cur_s, cur_e))
			cur_s, cur_e = s, e
	merged.append((cur_s, cur_e))
	return merged


def score_segments_tfidf(segments: List[Segment]) -> np.ndarray:
	texts = [s.text for s in segments]
	vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
	X = vectorizer.fit_transform(texts)
	# Simple importance: sum of TF-IDF weights per segment
	scores = np.asarray(X.sum(axis=1)).ravel()
	# Normalize
	if scores.max() > 0:
		scores = scores / scores.max()
	return scores


def select_segments_for_duration(segments: List[Segment], scores: np.ndarray, target_seconds: float) -> List[Tuple[float, float]]:
	# Greedy by score density (score per second)
	items = []
	for s, sc in zip(segments, scores):
		dur = max(1e-6, s.end - s.start)
		items.append((sc / dur, sc, s))
	items.sort(key=lambda x: (-x[0], -x[1]))

	selected: List[Tuple[float, float]] = []
	total = 0.0
	for _, _, seg in items:
		if total >= target_seconds:
			break
		selected.append((seg.start, seg.end))
		total += (seg.end - seg.start)
	# Merge nearby to reduce cuts
	selected = merge_adjacent_ranges(selected, max_gap=0.8)
	# If still over target, trim tail
	cur_total = sum(e - s for s, e in selected)
	if cur_total > target_seconds and selected:
		over = cur_total - target_seconds
		# Trim from the end of the last range
		s, e = selected[-1]
		selected[-1] = (s, max(s, e - over))
		# Drop zero-length if any
		selected = [(s, e) for s, e in selected if e - s > 0.05]
	return selected


def write_srt_for_summary(all_segments: List[Segment], chosen_ranges: List[Tuple[float, float]], srt_path: Path) -> None:
	# Build captions mapped into the concatenated timeline
	entries = []  # (start_new, end_new, text)
	cursor = 0.0
	for clip_start, clip_end in chosen_ranges:
		for seg in all_segments:
			if seg.end <= clip_start or seg.start >= clip_end:
				continue
			# Intersection with clip
			seg_s = max(seg.start, clip_start)
			seg_e = min(seg.end, clip_end)
			if seg_e - seg_s <= 0.05:
				continue
			start_new = cursor + (seg_s - clip_start)
			end_new = cursor + (seg_e - clip_start)
			entries.append((start_new, end_new, seg.text))
		cursor += (clip_end - clip_start)

	with open(srt_path, "w", encoding="utf-8") as f:
		for i, (s, e, text) in enumerate(entries, start=1):
			f.write(f"{i}\n")
			f.write(f"{format_ts(s)} --> {format_ts(e)}\n")
			f.write(f"{text}\n\n")


def format_ts(t: float) -> str:
	# SRT timestamp: HH:MM:SS,mmm
	ms = int(round(t * 1000))
	sec, ms = divmod(ms, 1000)
	m, s = divmod(sec, 60)
	h, m = divmod(m, 60)
	return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def cut_clip(input_video: Path, start: float, end: float, out_path: Path) -> None:
	duration = max(0.05, end - start)
	run_ffmpeg([
		"-ss", f"{start:.3f}",
		"-i", str(input_video),
		"-t", f"{duration:.3f}",
		"-analyzeduration", "0",
		"-probesize", "32M",
		"-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
		"-c:a", "aac", "-b:a", "128k",
		str(out_path),
	])


def concat_clips_with_demuxer(clips: List[Path], out_path: Path) -> None:
	with tempfile.TemporaryDirectory() as td:
		list_path = Path(td) / "list.txt"
		with open(list_path, "w", encoding="utf-8") as f:
			for clip in clips:
				f.write(f"file '{clip.as_posix()}'\n")
		run_ffmpeg(["-f", "concat", "-safe", "0", "-i", str(list_path), "-c", "copy", str(out_path)])


def burn_subtitles(input_video: Path, srt_path: Path, out_path: Path) -> None:
	# On Windows, escape the drive-letter colon in filter args. Use filename= for clarity.
	resolved = srt_path.resolve()
	# Use forward slashes then escape the drive-letter colon (e.g., C\:/...)
	sub_posix = resolved.as_posix().replace(":", r"\:")
	filter_arg = f"subtitles=filename='{sub_posix}'"
	run_ffmpeg([
		"-i", str(input_video),
		"-vf", filter_arg,
		"-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
		"-c:a", "aac", "-b:a", "160k",
		str(out_path),
	])


def main() -> None:
	parser = argparse.ArgumentParser(description="AI video summarizer using ffmpeg and Whisper")
	parser.add_argument("input", help="Path to input video")
	parser.add_argument("--out", dest="out_dir", default="out", help="Output directory")
	parser.add_argument("--target", type=float, default=60.0, help="Target summary length in seconds")
	parser.add_argument("--model", default="base", help="Whisper model size, e.g., tiny, base, small, medium, large")
	parser.add_argument("--device", default="", help="Torch device, e.g., cpu or cuda (auto if empty)")
	parser.add_argument("--burn", action="store_true", help="Burn captions into the final summary video")
	parser.add_argument("--segments-json", default="", help="Path to JSON list of segments to bypass Whisper (testing)")
	args = parser.parse_args()

	input_video = Path(args.input).resolve()
	out_dir = Path(args.out_dir).resolve()
	out_dir.mkdir(parents=True, exist_ok=True)

	if shutil.which("ffmpeg") is None:
		print("Error: ffmpeg is not installed or not in PATH.", file=sys.stderr)
		sys.exit(1)

	with tempfile.TemporaryDirectory() as td:
		td_path = Path(td)
		audio_wav = td_path / "audio.wav"
		print("Extracting audio...")
		extract_audio(input_video, audio_wav)

		if args.segments_json:
			print("Loading segments from JSON (testing mode)...")
			segments = load_segments_from_json(Path(args.segments_json))
		else:
			print("Transcribing with Whisper (this may take a while)...")
			segments = transcribe_audio(audio_wav, args.model, args.device)

		if not segments:
			print("No speech segments found.")
			sys.exit(1)

		print("Scoring segments and selecting highlights...")
		scores = score_segments_tfidf(segments)
		chosen_ranges = select_segments_for_duration(segments, scores, args.target)
		if not chosen_ranges:
			print("No ranges selected.")
			sys.exit(1)

		clips_dir = td_path / "clips"
		clips_dir.mkdir(parents=True, exist_ok=True)
		clip_paths: List[Path] = []
		print("Cutting clips...")
		for i, (s, e) in enumerate(tqdm(chosen_ranges)):
			clip_path = clips_dir / f"clip_{i:03d}.mp4"
			cut_clip(input_video, s, e, clip_path)
			clip_paths.append(clip_path)

		print("Concatenating clips...")
		summary_path = out_dir / f"{input_video.stem}_summary.mp4"
		concat_clips_with_demuxer(clip_paths, summary_path)

		srt_path = out_dir / f"{input_video.stem}_summary.srt"
		print("Writing captions (SRT)...")
		write_srt_for_summary(segments, chosen_ranges, srt_path)

		if args.burn:
			print("Burning captions into video...")
			burned_path = out_dir / f"{input_video.stem}_summary_captions.mp4"
			burn_subtitles(summary_path, srt_path, burned_path)
			print(f"Done. Output: {burned_path}")
		else:
			print(f"Done. Outputs: {summary_path}, {srt_path}")


if __name__ == "__main__":
	main()
