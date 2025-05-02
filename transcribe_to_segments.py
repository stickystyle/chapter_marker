import argparse
import json
from tqdm import tqdm
import sys
import mlx_whisper


def transcribe_with_mlx(audio_file: str):
    """
    Transcribe using mlx-whisper with word-level timestamps.
    Returns: list of segments (dicts with start, end, text, and words)
    """
    result = mlx_whisper.transcribe(audio_file, word_timestamps=True)
    return result["segments"]


def transcribe_to_jsonl(audio_file: str, output_file: str):
    """
    Transcribe the audio file and write segments to a JSONL file.
    Uses mlx-whisper with word-level timestamps.
    """
    print("Transcribing with mlx-whisper...")
    segments = transcribe_with_mlx(audio_file)

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
    parser = argparse.ArgumentParser(description="Transcribe audio to JSONL segments using mlx-whisper with word-level timestamps.")
    parser.add_argument('audio_file', help='Path to the audio file (e.g., test.m4a)')
    parser.add_argument('-o', '--output', default='segments.jsonl', help='Output JSONL file (default: segments.jsonl)')
    args = parser.parse_args()

    transcribe_to_jsonl(
        args.audio_file,
        args.output
    )


if __name__ == "__main__":
    main() 