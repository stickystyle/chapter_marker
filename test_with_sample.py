#!/usr/bin/env python3

import json
from detect_chapters_from_segments import ChapterDetector

def main():
    """
    Test chapter detection with sample segments containing chapter markers.
    """
    # Sample segments with chapter markers
    sample_segments = [
        {
            "id": 0,
            "text": "Chapter 1: The Beginning",
            "start": 10.5,
            "end": 12.3
        },
        {
            "id": 1,
            "text": "This is a regular segment with no chapter markers.",
            "start": 15.7,
            "end": 18.2
        },
        {
            "id": 2,
            "text": "29 seconds later, Ford and Arthur were rescued. Chapter 9",
            "start": 25.1,
            "end": 30.4
        },
        {
            "id": 3,
            "text": "Chapter 11 - The Restaurant at the End of the Universe",
            "start": 35.8,
            "end": 40.2
        }
    ]
    
    # Save sample segments to file
    with open("sample_segments.json", "w") as f:
        json.dump({"segments": sample_segments}, f)
    
    print("Created sample segments JSON file")
    
    # Initialize detector with proper model name and verbose output
    detector = ChapterDetector(
        use_mlx=True, 
        verbose=True,
        model_name="mlx-community/Llama-3.1-8B-Instruct-4bit"  # Specify the 8B model explicitly
    )
    
    # Ensure model is loaded
    if detector.use_mlx:
        print("Initializing MLX model...")
        if not detector.initialize_mlx_model():
            print("Failed to initialize MLX model, exiting")
            return
    
    # Process segments
    results = detector.process_segments(sample_segments, "sample_chapters.txt")
    
    # Display results
    print("\nDetected chapter markers:")
    for result in results:
        print(result)

if __name__ == "__main__":
    main() 