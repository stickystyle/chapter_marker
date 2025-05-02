# Audiobook Chapter Marker

A Python script that extracts chapter markers from audiobooks using OpenAI's Whisper for transcription and Qwen 2.5 for intelligent chapter detection.

## Features

- Accurate audio transcription using OpenAI's Whisper model
- Intelligent chapter detection using Qwen 2.5 1.5B model
- Fallback pattern matching for reliability
- Efficient model caching to avoid redundant downloads
- Progress reporting during processing
- Support for various chapter marker formats
- Clean timestamp formatting

## Installation

1. Clone the repository
2. Install dependencies using `uv`:
```bash
uv pip install -r requirements.txt
```

## Dependencies

The project uses the following key dependencies:
- `openai-whisper`: Audio transcription
- `transformers`: Qwen 2.5 model for chapter detection
- `pytest`: Testing framework
- `tqdm`: Progress bars
- `numpy`: Required for model operations

## Usage

Basic usage:
```bash
python chapter_marker.py <audio_file> [-o output.txt] [--cache-dir /path/to/cache]
```

Arguments:
- `audio_file`: Path to the audio file to process
- `-o, --output`: Output file path (default: chapters.txt)
- `--cache-dir`: Directory to cache models (default: ~/.cache/whisper_chapter_marker)

Example:
```bash
python chapter_marker.py audiobook.m4a -o chapters.txt
```

## How It Works

1. **Audio Transcription**: Uses OpenAI's Whisper model to transcribe the audio file into text segments with timestamps.

2. **Chapter Detection**: Uses a two-stage approach:
   - Primary: Qwen 2.5 1.5B model analyzes text segments for chapter markers
   - Fallback: Pattern matching using regex for reliability if model fails

3. **Output Generation**: Generates a text file with timestamps and chapter titles in the format:
```
00:01.500 - Chapter 1
01:30.000 - Chapter 2
```

## Model Caching

The script implements efficient model caching:
- Models are cached in `~/.cache/whisper_chapter_marker` by default
- Custom cache directory can be specified via `--cache-dir`
- Checks for existing cached models before downloading
- Separate cache directories for Whisper and Qwen models

## Testing

The project uses test-driven development with both unit and integration tests:

```bash
# Run all tests
pytest

# Run only fast tests (skip integration tests)
pytest -m "not slow"
```

The test suite includes:
- Unit tests for core functionality
- Mocked tests for LLM interactions
- Integration test using test.m4a (contains known chapter marker at 01:25.740)

## Supported Chapter Formats

The script can detect various chapter marker formats:
- Standard numbers: "Chapter 1", "Part 2"
- Roman numerals: "Chapter I", "Book IV"
- Written numbers: "Chapter One", "Part Three"
- Single letters: "Chapter A", "Section B"
- With descriptions: "Chapter 1: The Beginning", "Part II - Introduction"

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 