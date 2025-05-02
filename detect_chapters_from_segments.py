import argparse
import json
from tqdm import tqdm
from pathlib import Path
import re
import math
import os
from mlx_lm import load, generate

# --- Chapter detection logic (adapted from chapter_marker.py) ---

class ChapterDetector:
    def __init__(self, cache_dir=None, use_mlx=True):
        self.cache_dir = cache_dir or str(Path.home() / ".cache" / "whisper_chapter_marker")
        self.use_mlx = use_mlx
        self.model = None
        self.tokenizer = None
        os.makedirs(self.cache_dir, exist_ok=True)
        # Comprehensive patterns for detecting chapter markers
        self.patterns = [
            # Standard chapter markers - using word boundaries to avoid partial matches
            r'^\s*(?:chapter|part|section|book)\b\s+(?:\d+|[ivxlcdm]+|\w+)',
            r'^\s*(?:chapter|part|section|book)\b\s+[A-Z](?:\s|$)',
            # Number words
            r'^\s*(?:chapter|part|section|book)\b\s+(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)',
            # Roman numerals for chapters
            r'^\s*(?:chapter|part|section|book)\b\s+(?:i{1,3}|iv|v|vi{1,3}|ix|x|xi{1,3}|xiv|xv|xvi{1,3}|xix|xx|xxi{1,3})\b',
            # Prefixes with numbers (e.g., "Chapter 1:", "Part II -")
            r'^\s*(?:chapter|part|section|book)\b\s+(?:\d+|[ivxlcdm]+|\w+)[\s:-]',
        ]
        # Phrases to exclude (false positives)
        self.exclude_patterns = [
            r'part of',
            r'chapter of',
            r'section of',
        ]

    def initialize_mlx_model(self):
        """Initialize the Mistral-7B model using MLX."""
        if self.model is None and self.use_mlx:
            try:
                print("Loading Mistral-7B-Instruct model...")
                self.model, self.tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")
                print("Model loaded successfully")
                return True
            except Exception as e:
                print(f"Warning: Could not initialize MLX model: {e}")
                print("Falling back to regex patterns")
                self.use_mlx = False
                return False
        return self.model is not None

    def is_chapter_marker_mlx(self, text: str) -> bool:
        """
        Check if text is a chapter marker using the MLX Mistral model.
        
        Args:
            text (str): The text to analyze
            
        Returns:
            bool: True if the text is a chapter marker, False otherwise
        """
        try:
            # Use Llama 3 official chat prompt format
            system_message = (
                "You are an expert at detecting chapter markers in audiobook transcripts. "
                "A chapter marker MUST:\n"
                "- Start at the very beginning of the text\n"
                "- Begin with exactly one of: Chapter, Part, Section, or Book (case-insensitive)\n"
                "- Be immediately followed by a number, Roman numeral, or number word (e.g., 1, II, One)\n"
                "- Optionally, a short title after a colon, dash, or period\n"
                "- NOT be a regular sentence, dialogue, or narrative\n"
                "Return ONLY the word 'true' or 'false' without any explanation.\n"
                "Valid chapter marker examples:\n"
                "- Chapter 1\n- Part II\n- Section 3: The Beginning\n- Book One - Introduction\n- Chapter 2.\n"
                "NOT chapter markers (return 'false' for these):\n"
                "- Many solutions were suggested for this problem, but most of these were largely concerned with the movements of small, green pieces of paper.\n"
                "- And then, one Thursday nearly 2,000 years after one man had been nailed to a tree for saying how great it would be to be nice to people for a change,\n"
                "- The thing that used to worry him most was the fact that people always used to ask him what he was looking so worried about\n"
                "- Mr\n- Thank you very much, said Mr\n- Today he was particularly nervous and worried because something had gone seriously wrong with his job,\n- Curiously enough, though he didn't know it, he was also a direct male\n"
            )
            prompt = (
                "<|begin_of_text|>"
                "<|start_header_id|>system<|end_header_id|>\n"
                f"{system_message}\n"
                "<|eot_id|>"
                "<|start_header_id|>user<|end_header_id|>\n"
                f"Is the following text a chapter marker?\n{text}\n"
                "<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>"
            )
            response = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=10,
                verbose=False
            )
            response_text = response.strip().lower()
            if "true" in response_text and "false" not in response_text:
                return True
            elif "false" in response_text and "true" not in response_text:
                return False
            # If response is ambiguous, fall back to regex
            return self.is_chapter_marker_regex(text)
        except Exception as e:
            print(f"Error using MLX model: {e}")
            self.use_mlx = False
            return self.is_chapter_marker_regex(text)

    def is_chapter_marker_regex(self, text: str) -> bool:
        """
        Check if the given text represents a chapter marker using comprehensive regex patterns.
        
        Args:
            text (str): The text to check
            
        Returns:
            bool: True if the text appears to be a chapter marker, False otherwise
        """
        text = text.strip()
        text_lower = text.lower()
        
        # First check if text contains any exclusion patterns
        for exclude in self.exclude_patterns:
            if re.search(exclude, text_lower):
                return False
        
        # Then check against all our patterns
        for pattern in self.patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
                
        return False

    def is_chapter_marker(self, text: str) -> bool:
        """
        Check if the given text is a chapter marker using MLX if available,
        falling back to regex patterns if MLX is unavailable or fails.
        
        Args:
            text (str): The text to check
            
        Returns:
            bool: True if the text appears to be a chapter marker, False otherwise
        """
        # Try to use MLX if enabled
        if self.use_mlx:
            if self.model is None:
                self.initialize_mlx_model()
                
            if self.model is not None:
                return self.is_chapter_marker_mlx(text)
        
        # Fall back to regex patterns
        return self.is_chapter_marker_regex(text)

    def extract_chapter_info(self, text: str) -> str:
        """
        Extract clean chapter title from text by removing any content after delimiters.
        
        Args:
            text (str): Raw text containing the chapter marker
            
        Returns:
            str: Clean chapter title
        """
        # First clean the text
        text = text.strip()
        
        # Then extract everything before common delimiters
        for delimiter in [':', '-', '–', '—', '.']:
            parts = text.split(delimiter, 1)
            if len(parts) > 1:
                text = parts[0]
                break
                
        return text.strip()

    def format_timestamp(self, seconds: float) -> str:
        """
        Convert seconds to a formatted timestamp string.
        
        Args:
            seconds (float): Time in seconds
            
        Returns:
            str: Formatted timestamp in either "MM:SS.mmm" or "HH:MM:SS.mmm" format
        """
        hours = math.floor(seconds / 3600)
        minutes = math.floor((seconds % 3600) / 60)
        seconds = seconds % 60
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
        return f"{minutes:02d}:{seconds:06.3f}"

# --- Main script ---

def main():
    parser = argparse.ArgumentParser(description="Detect chapter markers from segments.jsonl using MLX or regex patterns and output chapters.txt.")
    parser.add_argument('segments_file', help='Path to segments.jsonl')
    parser.add_argument('-o', '--output', default='chapters.txt', help='Output chapters file (default: chapters.txt)')
    parser.add_argument('--cache-dir', default=str(Path.home() / ".cache" / "whisper_chapter_marker"), help='Cache directory')
    parser.add_argument('--no-mlx', action='store_true', help='Disable MLX and use only regex patterns')
    parser.add_argument('--max-kv-cache', type=int, default=4096, help='Maximum size of the KV cache for MLX model')
    parser.add_argument('--compare', action='store_true', help='Compare MLX and regex results side by side')
    args = parser.parse_args()

    if args.compare:
        # Run with both methods and compare
        print(f"Running comparison between MLX and regex detection...")
        
        # First with MLX
        mlx_detector = ChapterDetector(cache_dir=args.cache_dir, use_mlx=True)
        mlx_chapters = []
        
        print(f"Processing segments with MLX...")
        with open(args.segments_file, 'r', encoding='utf-8') as f:
            segments = list(f)
            for line in tqdm(segments, desc="MLX analysis"):
                seg = json.loads(line)
                text = seg["text"].strip()
                if mlx_detector.is_chapter_marker(text):
                    chapter_title = mlx_detector.extract_chapter_info(text)
                    timestamp = mlx_detector.format_timestamp(seg["start"])
                    mlx_chapters.append((timestamp, chapter_title))
        
        # Then with regex only
        regex_detector = ChapterDetector(cache_dir=args.cache_dir, use_mlx=False)
        regex_chapters = []
        
        print(f"Processing segments with regex...")
        with open(args.segments_file, 'r', encoding='utf-8') as f:
            segments = list(f)
            for line in tqdm(segments, desc="Regex analysis"):
                seg = json.loads(line)
                text = seg["text"].strip()
                if regex_detector.is_chapter_marker(text):
                    chapter_title = regex_detector.extract_chapter_info(text)
                    timestamp = regex_detector.format_timestamp(seg["start"])
                    regex_chapters.append((timestamp, chapter_title))
        
        # Output to separate files
        mlx_output = args.output.replace('.txt', '_mlx.txt')
        regex_output = args.output.replace('.txt', '_regex.txt')
        
        with open(mlx_output, 'w', encoding='utf-8') as f:
            for timestamp, title in mlx_chapters:
                f.write(f"{timestamp} - {title}\n")
        
        with open(regex_output, 'w', encoding='utf-8') as f:
            for timestamp, title in regex_chapters:
                f.write(f"{timestamp} - {title}\n")
        
        # Print comparison summary
        mlx_only = set((t, c) for t, c in mlx_chapters) - set((t, c) for t, c in regex_chapters)
        regex_only = set((t, c) for t, c in regex_chapters) - set((t, c) for t, c in mlx_chapters)
        
        print(f"\nComparison results:")
        print(f"MLX found {len(mlx_chapters)} chapter markers, saved to {mlx_output}")
        print(f"Regex found {len(regex_chapters)} chapter markers, saved to {regex_output}")
        print(f"{len(set((t, c) for t, c in mlx_chapters) & set((t, c) for t, c in regex_chapters))} chapters found by both methods")
        
        if mlx_only:
            print(f"\nChapters found by MLX but not regex:")
            for t, c in sorted(mlx_only):
                print(f"  {t} - {c}")
        
        if regex_only:
            print(f"\nChapters found by regex but not MLX:")
            for t, c in sorted(regex_only):
                print(f"  {t} - {c}")
    
    else:
        # Normal processing with one method
        detector = ChapterDetector(cache_dir=args.cache_dir, use_mlx=not args.no_mlx)
        chapters = []

        print(f"Processing segments from {args.segments_file}...")
        with open(args.segments_file, 'r', encoding='utf-8') as f:
            segments = list(f)
            for line in tqdm(segments, desc="Analyzing segments"):
                seg = json.loads(line)
                text = seg["text"].strip()
                if detector.is_chapter_marker(text):
                    chapter_title = detector.extract_chapter_info(text)
                    timestamp = detector.format_timestamp(seg["start"])
                    print(f"Found chapter: {timestamp} - {chapter_title}")
                    chapters.append((timestamp, chapter_title))

        with open(args.output, 'w', encoding='utf-8') as f:
            for timestamp, title in chapters:
                f.write(f"{timestamp} - {title}\n")
        print(f"\nWrote {len(chapters)} chapter markers to {args.output}")

if __name__ == "__main__":
    main() 