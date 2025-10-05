from pathlib import Path

from summarize_video import format_ts, merge_adjacent_ranges


def test_format_ts_rounding_and_format():
	assert format_ts(0.0) == "00:00:00,000"
	assert format_ts(1.234) == "00:00:01,234"
	# Check rounding overflows properly
	assert format_ts(60.0) == "00:01:00,000"


def test_merge_adjacent_ranges():
	input_ranges = [(0.0, 1.0), (1.3, 2.0), (3.0, 4.0), (4.4, 4.6)]
	# With max_gap=0.5, first two merge, last two merge
	merged = merge_adjacent_ranges(input_ranges, max_gap=0.5)
	assert merged == [(0.0, 2.0), (3.0, 4.6)]
