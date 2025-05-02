import pytest
from detect_chapters_from_segments import ChapterDetector
import json
import tempfile
import os
from unittest.mock import patch, MagicMock

def test_is_chapter_marker_regex():
    """Test the identification of chapter markers in text using regex patterns."""
    # Create detector with MLX disabled
    detector = ChapterDetector(use_mlx=False)
    
    # Test various chapter formats using regex patterns
    assert detector.is_chapter_marker("Chapter 1")
    assert detector.is_chapter_marker("Chapter One")
    assert detector.is_chapter_marker("Chapter I")
    assert detector.is_chapter_marker("Part 1")
    assert detector.is_chapter_marker("Section 2")
    assert detector.is_chapter_marker("Book One")
    
    # Test cases that should not be identified as chapters
    assert not detector.is_chapter_marker("The chapter was interesting")
    assert not detector.is_chapter_marker("He wrote a chapter")
    assert not detector.is_chapter_marker("Random text")
    
    # Test exclusion patterns
    assert not detector.is_chapter_marker("part of the story")
    assert not detector.is_chapter_marker("chapter of this book")

def test_is_chapter_marker_special_formats():
    """Test additional chapter formats."""
    detector = ChapterDetector(use_mlx=False)
    
    # Test roman numerals
    assert detector.is_chapter_marker("Chapter IV")
    assert detector.is_chapter_marker("Chapter XVI")
    
    # Test with delimiters
    assert detector.is_chapter_marker("Chapter 3: The Beginning")
    assert detector.is_chapter_marker("Part 2 - Introduction")

@pytest.mark.parametrize(
    "text,expected,model_response",
    [
        ("Chapter 1", True, "true"),
        ("Part 2: Introduction", True, "true"),
        ("Random text", False, "false"),
        ("Chapter 3", True, "I'm not sure, but probably true"),  # Test fallback to regex when model is ambiguous
    ],
)
def test_is_chapter_marker_with_mlx(text, expected, model_response):
    """Test the identification of chapter markers using mocked MLX model."""
    with patch("mlx_lm.load", return_value=(MagicMock(), MagicMock())), \
         patch("mlx_lm.generate", return_value=model_response):
        
        detector = ChapterDetector(use_mlx=True)
        # Mock model initialization
        detector.model = MagicMock()
        detector.tokenizer = MagicMock()
        
        assert detector.is_chapter_marker(text) == expected

def test_extract_chapter_info():
    """Test extraction of chapter information from text."""
    detector = ChapterDetector(use_mlx=False)
    
    # Test various formats
    assert detector.extract_chapter_info("Chapter 1") == "Chapter 1"
    assert detector.extract_chapter_info("Chapter One") == "Chapter One"
    assert detector.extract_chapter_info("Part 1: The Beginning") == "Part 1"
    assert detector.extract_chapter_info("Section 2 - Introduction") == "Section 2"
    assert detector.extract_chapter_info("Chapter IV — The Journey") == "Chapter IV"

def test_format_timestamp():
    """Test timestamp formatting."""
    detector = ChapterDetector(use_mlx=False)
    
    assert detector.format_timestamp(1.5) == "00:01.500"
    assert detector.format_timestamp(61.7) == "01:01.700"
    assert detector.format_timestamp(3661.1) == "01:01:01.100"

def test_main_function():
    """Test the main function with mock input file."""
    # Create a temporary segments.jsonl file
    segments = [
        {"start": 10.5, "text": "This is normal text."},
        {"start": 60.0, "text": "Chapter 1: Introduction"},
        {"start": 120.0, "text": "This is normal text again."},
        {"start": 180.0, "text": "Chapter 2: The Middle"}
    ]
    
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.jsonl', delete=False) as segments_file, \
         tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False) as output_file:
        
        segments_path = segments_file.name
        output_path = output_file.name
        
        # Write test segments
        for segment in segments:
            segments_file.write(json.dumps(segment) + "\n")
        segments_file.flush()
        
        # Mock the ChapterDetector to avoid actual model loading
        with patch("detect_chapters_from_segments.ChapterDetector") as MockDetector:
            # Setup the mock detector behavior
            mock_instance = MockDetector.return_value
            mock_instance.is_chapter_marker.side_effect = lambda text: "Chapter" in text
            mock_instance.extract_chapter_info.side_effect = lambda text: text.split(":")[0].strip()
            mock_instance.format_timestamp.side_effect = lambda seconds: f"{int(seconds//60):02d}:{seconds%60:06.3f}"
            
            # Run the test with command line arguments
            with patch("sys.argv", ["detect_chapters_from_segments.py", segments_path, "-o", output_path, "--no-mlx"]):
                from detect_chapters_from_segments import main
                main()
            
            # Verify the output file
            with open(output_path, 'r') as f:
                lines = f.readlines()
            
            assert len(lines) == 2
            assert "01:00.000 - Chapter 1" in lines[0]
            assert "03:00.000 - Chapter 2" in lines[1]
    
    # Clean up the temporary files
    if os.path.exists(segments_path):
        os.unlink(segments_path)
    if os.path.exists(output_path):
        os.unlink(output_path) 