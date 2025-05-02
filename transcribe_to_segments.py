import argparse
import json
from tqdm import tqdm
import sys

# Try to import mlx_whisper
try:
    import mlx_whisper
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

# Try to import openai-whisper only if needed
if not HAS_MLX:
    try:
        import whisper
    except ImportError:
        whisper = None


def transcribe_with_mlx(audio_file: str):
    """
    Transcribe using mlx-whisper with word-level timestamps.
    Returns: list of segments (dicts with start, end, text, and words)
    """
    result = mlx_whisper.transcribe(audio_file, word_timestamps=True)
    return result["segments"]


def transcribe_with_whisper(audio_file: str):
    """
    Transcribe using openai-whisper with word-level timestamps.
    Returns: list of segments (dicts with start, end, text, and words)
    """
    if whisper is None:
        raise ImportError("Neither mlx-whisper nor openai-whisper is installed.")
    print("Loading Whisper model...")
    model = whisper.load_model("turbo")
    result = model.transcribe(
        audio_file,
        language="en",
        task="transcribe",
        verbose=False,
        word_timestamps=True
    )
    return result["segments"]


def transcribe_to_jsonl(audio_file: str, output_file: str, use_mlx: bool = False):
    """
    Transcribe the audio file and write segments to a JSONL file.
    Uses mlx-whisper if available and requested, otherwise falls back to openai-whisper.
    Always uses word-level timestamps.
    """
    if use_mlx:
        if not HAS_MLX:
            print("mlx-whisper is not installed. Please install it or run without --use-mlx.", file=sys.stderr)
            sys.exit(1)
        print("Transcribing with mlx-whisper...")
        segments = transcribe_with_mlx(audio_file)
    else:
        if HAS_MLX:
            print("Transcribing with mlx-whisper (auto-detected)...")
            segments = transcribe_with_mlx(audio_file)
        else:
            print("Transcribing with openai-whisper...")
            segments = transcribe_with_whisper(audio_file)

    print(f"Writing {len(segments)} segments to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for segment in tqdm(segments, desc="Writing segments"):
            out = {
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"].strip()
            }
            if "words" in segment:
                out["words"] = segment["words"]
            json.dump(out, f)
            f.write('\n')
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio to JSONL segments using Whisper or mlx-whisper (always with word-level timestamps).")
    parser.add_argument('audio_file', help='Path to the audio file (e.g., test.m4a)')
    parser.add_argument('-o', '--output', default='segments.jsonl', help='Output JSONL file (default: segments.jsonl)')
    parser.add_argument('--use-mlx', action='store_true', help='Force use of mlx-whisper (fail if not installed)')
    args = parser.parse_args()

    transcribe_to_jsonl(
        args.audio_file,
        args.output,
        use_mlx=args.use_mlx
    )


if __name__ == "__main__":
    main() 