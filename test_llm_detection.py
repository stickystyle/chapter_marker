#!/usr/bin/env python3

import sys
from detect_chapters_from_segments import ChapterDetector

def test_llm_detection(model_name="mlx-community/Llama-3.1-8B-Instruct-4bit"):
    """
    Test the improved LLM detection capabilities.
    
    Args:
        model_name (str): Name of the MLX model to use
    """
    # Test cases with expected results - all should be detected as chapter markers,
    # but only some are at the beginning
    test_cases = [
        # Text, is_chapter, at_beginning
        ("Chapter 1: The Beginning", True, True),
        ("Chapter 2", True, True),
        ("CHAPTER 3 - The Journey", True, True),
        ("Part I: Introduction", True, True),
        ("Section 5. Details", True, True),
        ("Book Two", True, True),
        ("Chapter Ten: The Final Battle", True, True),
        ("Chapter XV", True, True),
        ("29 seconds later, Ford and Arthur were rescued. Chapter 9", True, False),
        ("Mr Prosser said, Chapter 11 begins with a bulldozer scene", True, False),
        ("Chapter", False, False),
        ("This is not a chapter marker", False, False),
    ]
    
    # Initialize detector with MLX
    detector = ChapterDetector(use_mlx=True, verbose=True, model_name=model_name)
    
    print(f"Using model: {model_name}")
    
    # Ensure model is loaded
    if not detector.initialize_mlx_model():
        print("Failed to initialize MLX model")
        return
    
    print("\n===== TESTING LLM CHAPTER MARKER DETECTION =====\n")
    
    for i, (text, expected_is_chapter, expected_at_beginning) in enumerate(test_cases):
        print(f"\nTest {i+1}: {text[:40]}{'...' if len(text) > 40 else ''}")
        
        # Get result from detector
        result = detector.is_chapter_marker_mlx(text)
        
        # Verify results
        is_chapter = result.get("is_chapter", False)
        at_beginning = result.get("at_beginning", False)
        
        print(f"Expected: is_chapter={expected_is_chapter}, at_beginning={expected_at_beginning}")
        print(f"Got:      is_chapter={is_chapter}, at_beginning={at_beginning}")
        
        if is_chapter == expected_is_chapter and at_beginning == expected_at_beginning:
            print("✅ PASSED")
        else:
            print("❌ FAILED")
    
    print("\n===== TEST COMPLETE =====")

if __name__ == "__main__":
    # Allow specifying a different model from command line
    model = "mlx-community/Llama-3.1-8B-Instruct-4bit"
    if len(sys.argv) > 1:
        model = sys.argv[1]
    test_llm_detection(model) 