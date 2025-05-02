import pytest
from chapter_marker import ChapterMarker, ChapterEntry
import os
import tempfile
from unittest.mock import patch, MagicMock

def test_is_chapter_marker():
    """Test the identification of chapter markers in text."""
    marker = ChapterMarker()
    
    # Test various chapter formats
    assert marker.is_chapter_marker("Chapter 1")
    assert marker.is_chapter_marker("Chapter One")
    assert marker.is_chapter_marker("Chapter I")
    assert marker.is_chapter_marker("Part 1")
    assert marker.is_chapter_marker("Section 2")
    assert marker.is_chapter_marker("Book One")
    
    # Test cases that should not be identified as chapters
    assert not marker.is_chapter_marker("The chapter was interesting")
    assert not marker.is_chapter_marker("He wrote a chapter")
    assert not marker.is_chapter_marker("Random text")

@pytest.mark.parametrize(
    "text,expected,model_response",
    [
        ("Chapter 1", True, "true"),
        ("Part 2: Introduction", True, "true"),
        ("Random text", False, "false"),
        ("Chapter 3", True, "I'm not sure, but probably true"),  # Test fallback to regex when model is ambiguous
    ],
)
def test_is_chapter_marker_with_direct_llm(text, expected, model_response):
    """Test the identification of chapter markers using direct qwen2.5 model."""
    with patch("transformers.pipeline") as mock_pipeline:
        # Configure the mock to return a specific response
        mock_model = MagicMock()
        mock_model.return_value = model_response
        mock_pipeline.return_value = mock_model
        
        marker = ChapterMarker()
        marker.llm_model = mock_model  # Inject the mock model
        
        # Test with direct model
        assert marker.is_chapter_marker(text) == expected

def test_extract_chapter_info():
    """Test extraction of chapter information from text."""
    marker = ChapterMarker()
    
    # Test various formats
    assert marker.extract_chapter_info("Chapter 1") == "Chapter 1"
    assert marker.extract_chapter_info("Chapter One") == "Chapter One"
    assert marker.extract_chapter_info("Part 1: The Beginning") == "Part 1"
    assert marker.extract_chapter_info("Section 2 - Introduction") == "Section 2"

def test_format_timestamp():
    """Test timestamp formatting."""
    marker = ChapterMarker()
    
    assert marker.format_timestamp(1.5) == "00:01.500"
    assert marker.format_timestamp(61.7) == "01:01.700"
    assert marker.format_timestamp(3661.1) == "01:01:01.100"

def test_save_chapters_to_file():
    """Test saving chapter markers to a file."""
    marker = ChapterMarker()
    chapters = [
        ChapterEntry(timestamp="00:00.000", title="Chapter 1"),
        ChapterEntry(timestamp="01:30.500", title="Chapter 2"),
        ChapterEntry(timestamp="02:15.750", title="Part 1"),
        ChapterEntry(timestamp="03:45.250", title="Section 3")
    ]
    
    # Create a temporary file for testing
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt') as temp_file:
        temp_filename = temp_file.name
        
        try:
            # Save chapters to the temporary file
            marker.save_chapters_to_file(chapters, temp_filename)
            
            # Read the file contents
            with open(temp_filename, 'r') as f:
                lines = f.readlines()
            
            # Verify the contents
            assert len(lines) == len(chapters)
            assert lines[0].strip() == "00:00.000 - Chapter 1"
            assert lines[1].strip() == "01:30.500 - Chapter 2"
            assert lines[2].strip() == "02:15.750 - Part 1"
            assert lines[3].strip() == "03:45.250 - Section 3"
            
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)

def test_initialize_llm_model():
    """Test initialization of the qwen2.5 model."""
    # Create a mock that will be returned by pipeline
    mock_model = MagicMock()
    
    # Use the mock in the initialize_llm_model method
    with patch("chapter_marker.pipeline", return_value=mock_model) as mock_pipeline:
        marker = ChapterMarker()
        marker.initialize_llm_model()
        
        # Verify the model was initialized
        assert marker.llm_model is mock_model
        mock_pipeline.assert_called_once()

@pytest.mark.slow
def test_process_audio_file():
    """
    Integration test using a real audio file.
    This test is marked as 'slow' and should be run selectively.
    """
    if not os.path.exists("test.m4a"):
        pytest.skip("Test audio file not found")
        
    marker = ChapterMarker()
    chapters = marker.process_audio_file("test.m4a")
    
    # Check if we found the expected chapter marker
    expected_time = 85.74  # 01:25.740 in seconds
    chapter_found = False
    
    for chapter in chapters:
        # Convert timestamp to seconds for comparison
        timestamp_parts = chapter.timestamp.split(':')
        if len(timestamp_parts) == 2:
            # Format: MM:SS.mmm
            minutes, seconds = timestamp_parts
            timestamp_seconds = float(minutes) * 60 + float(seconds)
        else:
            # Format: HH:MM:SS.mmm
            hours, minutes, seconds = timestamp_parts
            timestamp_seconds = float(hours) * 3600 + float(minutes) * 60 + float(seconds)
            
        # Accept various formats of "Chapter 1" and allow for 2-second timing difference
        title_variations = ["Chapter I", "Chapter 1", "Chapter 1.", "CHAPTER I", "CHAPTER 1"]
        if (chapter.title.upper() in [t.upper() for t in title_variations] and 
            abs(timestamp_seconds - expected_time) < 2.0):
            chapter_found = True
            break
    
    assert chapter_found, f"Expected to find Chapter 1/I at approximately {expected_time} seconds" 