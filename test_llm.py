import json
import sys
from pathlib import Path
from detect_chapters_from_segments import ChapterDetector

def test_llm_chapter_detection(model_name="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"):
    # Initialize the LLM-based detector
    detector = ChapterDetector(use_mlx=True, verbose=True, model_name=model_name)
    
    print(f"Using model: {model_name}")
    
    # Test cases that need to be correctly identified
    test_cases = [
        {
            "name": "Chapter 9 at end of sentence",
            "text": "29 seconds later, Ford and Arthur were rescued. Chapter 9",
            "should_detect": True
        },
        {
            "name": "Chapter 11 at end of sentence",
            "text": "realised that the one thing they really couldn't stand was a smartass. Chapter 11",
            "should_detect": True
        },
        {
            "name": "Simple Chapter marker",
            "text": "Chapter 1",
            "should_detect": True
        },
        {
            "name": "Non-chapter text",
            "text": "This is just some regular text without any chapter markers.",
            "should_detect": False
        }
    ]
    
    print("Testing LLM chapter detection...\n")
    
    # Initialize the model first to avoid repeating this for each test
    detector.initialize_mlx_model()
    
    for case in test_cases:
        print(f"Testing: {case['name']}")
        print(f"Text: \"{case['text']}\"")
        
        # Create a mock segment with just the text
        mock_segment = {
            "text": case["text"],
            "words": []  # Empty words array as we're just testing the LLM text detection
        }
        
        # Check if the LLM considers this a chapter marker directly
        result = detector.is_chapter_marker(case["text"])
        is_chapter = result["is_chapter"]
        at_beginning = result["at_beginning"]
        
        # Record the result
        print(f"LLM response: is_chapter={is_chapter}, at_beginning={at_beginning}")
        print(f"Expected chapter detection: {case['should_detect']}")
        print(f"Result: {'✅ PASS' if is_chapter == case['should_detect'] else '❌ FAIL'}\n")

if __name__ == "__main__":
    model = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
    if len(sys.argv) > 1:
        model = sys.argv[1]
    test_llm_chapter_detection(model) 