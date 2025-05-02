# Audiobook Chapter Marker Detection

## Overview

This project provides a robust, LLM-powered tool for extracting chapter markers from audiobook transcriptions. It is designed for developers who need precise, timestamped chapter information from Whisper (or similar) transcript segment files. The detection logic leverages a large language model (Llama 3.1 8B via MLX) to identify chapter markers anywhere in a segment, and uses a sliding window approach to pinpoint the exact word and timestamp for each chapter marker.

## Features

- **LLM-Based Detection:** Uses a large language model (Llama 3.1 8B) to robustly identify chapter markers anywhere in a text segment, not just at the beginning.
- **Sliding Window Timestamping:** For chapter markers not at the beginning of a segment, a sliding window over the `words` array finds the precise word and timestamp where the marker starts.
- **No Regex/Pattern Fallback:** All chapter marker detection is performed by the LLM, not by regex or pattern matching.
- **Flexible Input:** Supports both JSON and JSONL formats for input segments.
- **Accurate Output:** Produces a text file with lines in the format: `HH:MM:SS.mmm - Chapter X: Title`.
- **Command-Line Interface:** Easy to use from the command line with options for model selection, output file, and verbosity.


## Installation & Setup

1. **Clone the repository** and set up a Python 3.10+ environment (virtualenv recommended).
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   - Key dependencies: `mlx-lm`, `mlx-whisper`, `tqdm`, `numpy`, `pytest`

3. **(Optional) Download or prepare your Whisper transcript segments** in JSON or JSONL format, with word-level timestamps.

## Usage

### 1. Transcribe Audio (Optional)
If you have an audio file and want to generate segments with word-level timestamps:
```bash
python transcribe_to_segments.py <audio_file> -o segments.jsonl
```
This uses `mlx-whisper` to produce a JSONL file where each line is a segment:
```json
{"start": 291.18, "end": 292.54, "text": "Chapter 1", "words": [{"word": " Chapter", "start": 291.18, "end": 291.86, ...}, ...]}
```

### 2. Detect Chapter Markers
Run the main detection script on your segments file:
```bash
python detect_chapters_from_segments.py <segments.jsonl or .json> [-o output.txt] [--no-mlx] [--model <model_name>] [-v]
```
- `-o output.txt` : Output file for chapter markers (default: `<input>_chapters.txt`)
- `--no-mlx`      : Disable MLX/LLM detection (not recommended)
- `--model`       : Specify MLX model (default: Llama 3.1 8B Instruct)
- `-v`            : Verbose output (for debugging)

#### Example
```bash
python detect_chapters_from_segments.py segments-chapter-only.jsonl -o chapters.txt
```

### Input Format
- **JSONL:** Each line is a segment with `start`, `end`, `text`, and `words` (word-level timestamps).
- **JSON:** Either an array of segments or an object with a `segments` field.

### Output Format
A text file with lines like:
```
04:51.180 - Chapter 1
30:45.200 - Chapter 2
...
```
Timestamps are precise to the word where the chapter marker begins.

## How It Works

### LLM-Based Chapter Detection
- The core logic is in `detect_chapters_from_segments.py`, primarily the `ChapterDetector` class.
- The method `is_chapter_marker_mlx` uses a Llama 3.1 8B model (via MLX) to analyze each segment's text and determine:
  - If a chapter marker is present (e.g., "Chapter 9", "Section II")
  - Whether the marker is at the beginning of the segment
- The LLM is prompted with strict rules and examples to ensure robust detection, even for edge cases.

### Sliding Window Timestamping
- If a chapter marker is not at the start of a segment, the code uses a sliding window (3, then 2 words) over the `words` array.
- For each window, it queries the LLM to check if the window contains a chapter marker at the beginning.
- The timestamp of the first word in the window is used as the precise chapter marker time.
- This ensures accurate alignment even when markers appear mid-sentence.


## Example

**Input segment (JSONL):**
```
{"start": 8042.62, "end": 8052.24, "text": "29 seconds later, Ford and Arthur were rescued. Chapter 9", ...}
```
**Output:**
```
02:14:11.100 - Chapter 9
```

## Testing & Validation
- The main workflow is validated by running the script on sample segment files and comparing the output to known-good chapter lists (see `test_chapters.txt`, `valid_chapters.txt`).
- You can use `pytest` for any additional tests you add.

## Extensibility
- The detection logic is modular and can be adapted to other LLMs or chapter marker conventions by modifying the prompt and extraction logic in `ChapterDetector`.
- Input/output formats are easily extensible for integration with other tools.

## Notes
- **Performance:** LLM-based detection is slower than regex, but far more robust to real-world audiobook transcript variations.
- **Dependencies:** Requires Apple Silicon (MLX) for fastest LLM inference, but can be adapted to other backends.


## License
MIT
