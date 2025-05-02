import argparse
import json
import re
import math
from pathlib import Path
from tqdm import tqdm

# Check if MLX is available
HAS_MLX = False
try:
    from mlx_lm import load, generate
    HAS_MLX = True
except ImportError:
    print("MLX LM not available. Install with: pip install mlx-lm")
    HAS_MLX = False


class ChapterDetector:
    """
    A class to detect chapter markers in transcribed audio segments
    using MLX or regex patterns.
    """
    
    def __init__(self, use_mlx=True, model_name=None, verbose=False):
        """
        Initialize the ChapterDetector.
        
        Args:
            use_mlx (bool): Whether to use MLX for chapter detection
            model_name (str, optional): Name of the MLX model to use
            verbose (bool): Whether to print verbose output
        """
        self.use_mlx = use_mlx and HAS_MLX
        self.model = None
        self.tokenizer = None
        self.model_name = model_name or "mlx-community/Llama-3.1-8B-Instruct-4bit"  # Default to 8B model
        self.verbose = verbose
        
        # Diagnostic output
        if verbose:
            print(f"MLX available: {HAS_MLX}")
            print(f"Using MLX: {self.use_mlx}")
            if self.use_mlx:
                print(f"Model: {self.model_name}")

    def initialize_mlx_model(self):
        """Initialize model using MLX."""
        if self.model is None and self.use_mlx:
            try:
                if self.verbose:
                    print(f"Loading model: {self.model_name}...")
                self.model, self.tokenizer = load(self.model_name)
                if self.verbose:
                    print("Model loaded successfully")
                return True
            except Exception as e:
                print(f"Error initializing MLX model: {e}")
                self.use_mlx = False
                return False
        return self.model is not None

    def is_chapter_marker_mlx(self, text: str) -> dict:
        """
        Check if text contains a chapter marker using the MLX model.
        
        Args:
            text (str): The text to analyze
            
        Returns:
            dict: {'is_chapter': bool, 'at_beginning': bool}
        """
        try:
            # Use Llama 3 official chat prompt format
            system_message = (
                "You are an expert at detecting chapter markers in audiobook transcripts. "
                "A chapter marker is defined as:\n"
                "- One of these words: Chapter, Part, Section, or Book (case-insensitive)\n"
                "- IMMEDIATELY followed by a number, Roman numeral, or number word (e.g., 1, II, One)\n"
                "- Optionally, a short title after a colon, dash, or period\n\n"
                "You need to check two things:\n"
                "1. Does the text contain a valid chapter marker anywhere?\n"
                "2. Is the chapter marker at the beginning of the text?\n\n"
                "STRICT RULES:\n"
                "- The word 'Chapter' by itself is NOT a valid chapter marker\n"
                "- A chapter marker is only considered at the beginning if it starts within the first 3 words\n"
                "- A chapter marker may appear in the middle or end of a sentence like '29 seconds later, Ford and Arthur were rescued. Chapter 9'\n"
                "- In the example above, 'Chapter 9' is a valid chapter marker but NOT at the beginning\n"
                "- You must check the position of the phrase in the text\n\n"
                "Return ONLY a valid JSON format with NO explanation, NO commentary, and NO description, just return the JSON object with two boolean fields: {\"is_chapter\": true/false, \"at_beginning\": true/false}\n\n"
                "Examples:\n"
                "- Input: \"Chapter 1\" → Output: {\"is_chapter\": true, \"at_beginning\": true}\n"
                "- Input: \"Section 5. Details\" → Output: {\"is_chapter\": true, \"at_beginning\": true}\n"
                "- Input: \"29 seconds later, Ford and Arthur were rescued. Chapter 9\" → Output: {\"is_chapter\": true, \"at_beginning\": false}\n"
                "- Input: \"Mr Prosser said, Chapter 11 begins with a bulldozer scene\" → Output: {\"is_chapter\": true, \"at_beginning\": false}\n"
                "- Input: \"Chapter\" → Output: {\"is_chapter\": false, \"at_beginning\": false}\n"
                "- Input: \"Mr Prosser said, 'Have you any idea how much damage that bulldozer would suffer if I just let it roll straight over you?'\" → Output: {\"is_chapter\": false, \"at_beginning\": false}\n"
            )
            prompt = (
                "<|begin_of_text|>"
                "<|start_header_id|>system<|end_header_id|>\n"
                f"{system_message}\n"
                "<|eot_id|>"
                "<|start_header_id|>user<|end_header_id|>\n"
                f"Analyze this text for chapter markers:\n{text}\n"
                "<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>"
            )
            response = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=100,
                verbose=False
            )
            
            # Parse the response as JSON
            response_text = response.strip()
            if self.verbose:
                print(f"LLM response: {response_text}")
                
            # Try different JSON parsing approaches:
            
            # 1. Try direct JSON parsing first
            try:
                # Check if response starts with a JSON object
                if response_text.startswith('{') and '}' in response_text:
                    json_str = response_text.split('}', 1)[0] + '}'
                    result = json.loads(json_str)
                    if "is_chapter" in result and "at_beginning" in result:
                        return result
            except:
                pass
                
            # 2. Try to find a JSON object anywhere in the text
            try:
                json_match = re.search(r'\{.*"is_chapter".*"at_beginning".*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    # Clean up any internal line breaks or extra spaces
                    json_str = re.sub(r'\s+', ' ', json_str)
                    result = json.loads(json_str)
                    if "is_chapter" in result and "at_beginning" in result:
                        return result
            except:
                pass
                
            # 3. Parse "true" and "false" occurrences
            is_chapter = False
            at_beginning = False
            
            # Look for explicit true/false statements
            if re.search(r'is_chapter"?\s*[:=]\s*true', response_text, re.IGNORECASE):
                is_chapter = True
            if re.search(r'at_beginning"?\s*[:=]\s*true', response_text, re.IGNORECASE):
                at_beginning = True
                
            # Return manual parsing result
            return {"is_chapter": is_chapter, "at_beginning": at_beginning}
                
        except Exception as e:
            if self.verbose:
                print(f"Error using MLX model: {e}")
            return {"is_chapter": False, "at_beginning": False}

    def is_chapter_marker(self, text: str) -> dict:
        """
        Check if the given text is a chapter marker using MLX only.
        
        Args:
            text (str): The text to check
            
        Returns:
            dict: {'is_chapter': bool, 'at_beginning': bool}
        """
        # Initialize the model if needed
        if self.use_mlx and self.model is None:
            self.initialize_mlx_model()
                
        if self.model is not None:
            return self.is_chapter_marker_mlx(text)
        else:
            # If MLX isn't available, return default response
            if self.verbose:
                print("MLX model not available")
            return {"is_chapter": False, "at_beginning": False}

    def extract_chapter_info(self, text: str) -> str:
        """
        Extract chapter information from the given text.
        
        Args:
            text (str): The text to extract chapter information from
            
        Returns:
            str: The extracted chapter information or an empty string if none found
        """
        # Check if it contains a chapter marker
        result = self.is_chapter_marker(text)
        
        if result["is_chapter"]:
            # Find chapter text using regex
            chapter_pattern = r'\b(chapter|part|section|book)\s+(\d+|[ivxlcdm]+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)(?:\s*[-:.]?\s*(.+))?'
            match = re.search(chapter_pattern, text, re.IGNORECASE)
            
            if match:
                prefix = match.group(1).capitalize()
                number = match.group(2)
                
                # Just return the chapter prefix and number without any extra text
                return f"{prefix} {number}"
        
        return ""
        
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
        
    def find_precise_chapter_timestamp(self, segment):
        """
        Use sliding window approach to find the exact timestamp of a chapter marker
        within a segment when it's not at the beginning.
        
        Args:
            segment (dict): The segment containing words with timestamps
            
        Returns:
            float: The precise timestamp of the chapter marker, or None if not found
        """
        if not "words" in segment or not segment["words"]:
            if self.verbose:
                print("No words data in segment")
            return None
            
        words = segment["words"]
        word_texts = [w["word"].strip() for w in words]
        
        # First try with window size 3
        if self.verbose:
            print("Trying to find chapter marker with window size 3")
        
        for i in range(len(words) - 2):
            # Extract window of 3 words
            window_text = " ".join(word_texts[i:i+3])
            
            # Check if this window contains a chapter marker at the beginning
            result = self.is_chapter_marker_mlx(window_text)
            
            if result["is_chapter"] and result["at_beginning"]:
                # Found the chapter marker at this position
                if self.verbose:
                    print(f"Found chapter marker in window: '{window_text}'")
                return words[i]["start"]
        
        # If not found with window size 3, try with window size 2
        if self.verbose:
            print("Trying with window size 2")
            
        for i in range(len(words) - 1):
            # Extract window of 2 words
            window_text = " ".join(word_texts[i:i+2])
            
            # Check if this window contains a chapter marker at the beginning
            result = self.is_chapter_marker_mlx(window_text)
            
            if result["is_chapter"] and result["at_beginning"]:
                # Found the chapter marker at this position
                if self.verbose:
                    print(f"Found chapter marker in window: '{window_text}'")
                return words[i]["start"]
        
        # If still not found, try individual words that match chapter/part keywords
        for i, word in enumerate(word_texts):
            if any(marker in word.lower() for marker in ["chapter", "part", "section", "book"]):
                if i+1 < len(words):  # Make sure there's a next word
                    # Check if the next word looks like a number
                    next_word = word_texts[i+1]
                    if any(char.isdigit() for char in next_word):
                        if self.verbose:
                            print(f"Found chapter marker with individual words: '{word} {next_word}'")
                        return words[i]["start"]
        
        # If we get here, we couldn't find a precise timestamp
        if self.verbose:
            print("Could not find precise chapter timestamp in segment")
        return None

    def process_segment(self, segment, result_list, tqdm_bar=None):
        """
        Process a single segment of transcribed audio.
        
        Args:
            segment: The segment to process
            result_list: List to append results to
            tqdm_bar: Optional progress bar
        """
        if self.verbose:
            print(f"Processing segment: {segment['text']}")
        
        # Check if the segment text contains a chapter marker
        marker_result = self.is_chapter_marker(segment["text"])
        
        if marker_result["is_chapter"]:
            # Default timestamp (from start of segment)
            timestamp = segment["start"]
            chapter_info = self.extract_chapter_info(segment["text"])
            
            # If the chapter marker is not at the beginning,
            # try to find the precise timestamp using sliding window
            if not marker_result["at_beginning"]:
                if self.verbose:
                    print("Chapter marker not at beginning, finding precise timestamp...")
                precise_timestamp = self.find_precise_chapter_timestamp(segment)
                if precise_timestamp is not None:
                    timestamp = precise_timestamp
                    if self.verbose:
                        print(f"Using precise timestamp: {self.format_timestamp(timestamp)}")
            
            if chapter_info:
                formatted_time = self.format_timestamp(timestamp)
                
                # Add to results
                result = {
                    "timestamp": formatted_time,
                    "raw_time": timestamp,
                    "chapter": chapter_info,
                    "at_beginning": marker_result["at_beginning"]
                }
                
                result_list.append(result)
                
                if self.verbose:
                    print(f"Found chapter marker: {chapter_info} at {formatted_time}")
        
        # Update progress bar if provided
        if tqdm_bar is not None:
            tqdm_bar.update(1)

    def process_segments(self, segments, output_file=None):
        """
        Process all segments from transcribed audio and extract chapter markers.
        
        Args:
            segments (list): List of segments from whisper transcription
            output_file (str, optional): Path to save chapter markers
            
        Returns:
            list: List of chapter markers with timestamps
        """
        from tqdm import tqdm
        
        results = []
        
        # Setup progress bar
        print(f"Processing {len(segments)} segments for chapter markers...")
        tqdm_bar = tqdm(total=len(segments))
        
        # Process each segment
        for segment in segments:
            self.process_segment(segment, results, tqdm_bar)
            
        tqdm_bar.close()
        
        # Sort results by timestamp
        results.sort(key=lambda x: x["raw_time"])
        
        # Format results for output
        formatted_results = []
        for result in results:
            formatted_results.append(f"{result['timestamp']} - {result['chapter']}")
            
        # Save to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                for line in formatted_results:
                    f.write(line + '\n')
            print(f"Chapter markers saved to {output_file}")
            
        return formatted_results

# --- Main script ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect chapter markers from Whisper transcribed audio segments")
    parser.add_argument("segments_file", help="JSON or JSONL file containing Whisper transcription segments")
    parser.add_argument("-o", "--output", help="Output file to save chapter markers")
    parser.add_argument("--no-mlx", action="store_true", help="Disable MLX model usage")
    parser.add_argument("--model", default="mlx-community/Llama-3.1-8B-Instruct-4bit", 
                      help="MLX model to use (default: mlx-community/Llama-3.1-8B-Instruct-4bit)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    
    # Load segments - handle both JSON and JSONL formats
    try:
        segments = []
        file_extension = Path(args.segments_file).suffix.lower()
        
        # Check if it's a JSONL file
        if file_extension == '.jsonl':
            with open(args.segments_file, 'r') as f:
                for line in f:
                    if line.strip():  # Skip empty lines
                        segment = json.loads(line)
                        segments.append(segment)
            print(f"Loaded {len(segments)} segments from JSONL file: {args.segments_file}")
        else:
            # Try to load as regular JSON
            with open(args.segments_file, 'r') as f:
                data = json.load(f)
                # Check if it has a "segments" field
                if "segments" in data:
                    segments = data["segments"]
                # If not, it might be an array directly
                elif isinstance(data, list):
                    segments = data
                else:
                    raise ValueError("JSON file does not contain segments array")
            print(f"Loaded {len(segments)} segments from JSON file: {args.segments_file}")
    except Exception as e:
        print(f"Error loading segments file: {e}")
        exit(1)
    
    # Initialize detector
    detector = ChapterDetector(
        use_mlx=not args.no_mlx,
        verbose=args.verbose,
        model_name=args.model
    )
    
    # Process segments and output results
    output_file = args.output if args.output else f"{Path(args.segments_file).stem}_chapters.txt"
    detector.process_segments(segments, output_file) 