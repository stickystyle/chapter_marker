import whisper
import re
from dataclasses import dataclass
from typing import List, Optional, Callable
import math
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import json
from tqdm import tqdm
import argparse
import sys
import os
from pathlib import Path

@dataclass
class ChapterEntry:
    """Represents a chapter marker with its timestamp and title."""
    timestamp: str
    title: str

class ChapterMarker:
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize a ChapterMarker instance.

        Args:
            cache_dir (Optional[str]): Directory to cache models. If None, defaults to ~/.cache/whisper_chapter_marker.
                                     The directory will be created if it doesn't exist.
        
        The class uses two models:
        1. Whisper model for audio transcription
        2. Qwen 2.5 1.5B model for intelligent chapter marker detection
        """
        self.model = None
        self.llm_model = None
        self.cache_dir = cache_dir or str(Path.home() / ".cache" / "whisper_chapter_marker")
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        # Improved patterns for better matching
        self.basic_patterns = [
            r'^(?:chapter|part|section|book)\s+(?:\d+|[ivxlcdm]+|\w+)',  # Matches any word/number after chapter/part/etc
            r'^(?:chapter|part|section|book)\s+[A-Z](?:\s|$)',  # Matches single letters like "Chapter A"
            r'^(?:chapter|part|section|book)\s+(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)'  # Common number words
        ]

    def initialize_llm_model(self):
        """
        Initialize the Qwen 2.5 1.5B model for chapter detection.

        This method:
        1. Sets up the model cache directory
        2. Checks if the model is already downloaded
        3. Downloads the model if necessary
        4. Configures the model with appropriate parameters for chapter detection

        The model is configured with:
        - Low temperature (0.01) for deterministic responses
        - Small max_new_tokens (10) since we only need true/false responses
        - Repetition penalty to prevent redundant output

        Raises:
            Exception: If model initialization fails
        """
        if self.llm_model is None:
            model_name = "Qwen/Qwen2.5-1.5B"
            print("Loading Qwen 2.5 model...")
            
            # Set model cache directory
            model_cache_dir = os.path.join(self.cache_dir, "qwen2.5")
            os.makedirs(model_cache_dir, exist_ok=True)
            
            # Check if model is already downloaded
            if os.path.exists(model_cache_dir) and any(f.startswith("config.json") for f in os.listdir(model_cache_dir)):
                print("Using cached model...")
            else:
                print("Downloading model (this may take a while)...")
            
            # Load tokenizer and model with explicit caching
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_cache_dir, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=model_cache_dir, trust_remote_code=True)
            
            self.llm_model = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=10,  # We only need a short response
                temperature=0.01,   # Low temperature for more deterministic responses
                do_sample=True,     # Required when using temperature
                repetition_penalty=1.1,
            )
            print("Qwen 2.5 model loaded.")

    def is_chapter_marker(self, text: str) -> bool:
        """
        Check if the given text contains a chapter marker using Qwen 2.5 1.5B model.

        This method uses a two-stage approach:
        1. First attempts to use the Qwen model for intelligent detection
        2. Falls back to regex patterns if the model fails or gives ambiguous responses

        Args:
            text (str): The text to analyze for chapter markers

        Returns:
            bool: True if the text is identified as a chapter marker, False otherwise

        Note:
            The method is resilient to model failures and will fall back to pattern matching
            if the model is unavailable or returns unclear results.
        """
        text = text.strip()
        
        # Try qwen2.5 model first
        if self.llm_model is None:
            try:
                self.initialize_llm_model()
            except Exception as e:
                print(f"Warning: Could not initialize LLM model: {e}")
                # Continue with pattern matching if model initialization fails
        
        if self.llm_model is not None:
            try:
                prompt = f"""Analyze this text: "{text}"
                Is this text a chapter marker (like "Chapter X", "Part Y", "Section Z", "Book W" where X/Y/Z/W can be any number, letter, or numeral)?
                Consider all formats of numbers (digits, words, roman numerals).
                Return ONLY the word 'true' or 'false' without any explanation."""
                
                response = self.llm_model(prompt)
                if isinstance(response, list):
                    result = response[0]["generated_text"]
                else:
                    result = response["generated_text"]
                
                # Extract just the generated part after the prompt
                generated_part = result[len(prompt):].strip().lower()
                
                # Check if the model gave a clear true/false response
                if "true" in generated_part and "false" not in generated_part:
                    return True
                elif "false" in generated_part and "true" not in generated_part:
                    return False
                # If response is ambiguous, fall back to pattern matching
            except Exception as e:
                print(f"Warning: Error using LLM model: {e}")
                # Continue to pattern matching if model fails
        
        # Fallback to pattern matching
        text_lower = text.lower()
        return any(re.search(pattern, text_lower, re.IGNORECASE) for pattern in self.basic_patterns)

    def extract_chapter_info(self, text: str) -> str:
        """
        Extract the clean chapter title from the text.

        This method processes the raw text to extract just the chapter marker part,
        removing any additional descriptions or content that follows common delimiters.

        Args:
            text (str): Raw text containing the chapter marker

        Returns:
            str: Clean chapter title with delimiters and extra content removed

        Example:
            >>> extract_chapter_info("Chapter 1: The Beginning")
            "Chapter 1"
            >>> extract_chapter_info("Part 2 - Introduction")
            "Part 2"
        """
        # Split on common delimiters and take the first part
        for delimiter in [':', '-', '–']:
            text = text.split(delimiter)[0]
        return text.strip()

    def format_timestamp(self, seconds: float) -> str:
        """
        Convert seconds to a formatted timestamp string.

        Args:
            seconds (float): Time in seconds

        Returns:
            str: Formatted timestamp in either "MM:SS.mmm" or "HH:MM:SS.mmm" format,
                 depending on whether hours are present

        Example:
            >>> format_timestamp(61.5)
            "01:01.500"
            >>> format_timestamp(3661.1)
            "01:01:01.100"
        """
        hours = math.floor(seconds / 3600)
        minutes = math.floor((seconds % 3600) / 60)
        seconds = seconds % 60
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
        return f"{minutes:02d}:{seconds:06.3f}"

    def process_audio_file(self, audio_file: str, progress_callback: Optional[Callable[[float], None]] = None) -> List[ChapterEntry]:
        """
        Process an audio file to extract chapter markers with their timestamps.

        This method:
        1. Loads the Whisper model if not already loaded
        2. Transcribes the audio file
        3. Analyzes each segment for chapter markers
        4. Returns a list of found chapters with their timestamps

        Args:
            audio_file (str): Path to the audio file to process
            progress_callback (Optional[Callable[[float], None]]): Optional callback function
                to report progress from 0.0 to 1.0

        Returns:
            List[ChapterEntry]: List of found chapters, each containing a timestamp and title

        Raises:
            Exception: If audio file processing fails
        """
        # Load the model only when needed
        if self.model is None:
            print("Loading Whisper model...")
            self.model = whisper.load_model("turbo")
        
        # Transcribe the audio file with progress reporting
        print("\nTranscribing audio file...")
        result = self.model.transcribe(
            audio_file,
            language="en",
            task="transcribe",
            verbose=False,  # We'll handle our own progress reporting
            condition_on_previous_text=False
        )
        
        chapters = []
        
        # Process each segment with progress bar
        print("\nAnalyzing segments for chapter markers:")
        segments = result["segments"]
        for segment in tqdm(segments, desc="Processing segments"):
            text = segment["text"].strip()
            if self.is_chapter_marker(text):
                chapter_title = self.extract_chapter_info(text)
                timestamp = self.format_timestamp(segment["start"])
                print(f"\nFound chapter marker: {chapter_title} at {timestamp}")
                chapters.append(ChapterEntry(timestamp=timestamp, title=chapter_title))
        
        print(f"\nFound {len(chapters)} chapter markers.")
        return chapters

    def save_chapters_to_file(self, chapters: List[ChapterEntry], output_file: str):
        """
        Save the found chapter markers to a text file.

        Args:
            chapters (List[ChapterEntry]): List of chapter entries to save
            output_file (str): Path to the output file

        The output format is:
        timestamp - chapter_title
        For example:
        00:01.500 - Chapter 1
        01:30.000 - Chapter 2

        Raises:
            IOError: If writing to the output file fails
        """
        with open(output_file, 'w') as f:
            for chapter in chapters:
                f.write(f"{chapter.timestamp} - {chapter.title}\n")
        print(f"\nChapter markers saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Extract chapter markers from an audiobook.')
    parser.add_argument('audio_file', help='Path to the audio file')
    parser.add_argument('-o', '--output', help='Output file path (default: chapters.txt)', default='chapters.txt')
    parser.add_argument('--cache-dir', help='Directory to cache models (default: ~/.cache/whisper_chapter_marker)', 
                       default=str(Path.home() / ".cache" / "whisper_chapter_marker"))
    
    args = parser.parse_args()
    
    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file '{args.audio_file}' not found.", file=sys.stderr)
        sys.exit(1)
    
    try:
        marker = ChapterMarker(cache_dir=args.cache_dir)
        chapters = marker.process_audio_file(args.audio_file)
        marker.save_chapters_to_file(chapters, args.output)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main() 