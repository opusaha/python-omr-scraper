import sys
import os
from omr_analyzer import OMRAnalyzer

def main():
    # Check if image path is provided
    if len(sys.argv) < 2:
        return
    
    image_path = sys.argv[1]
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"❌ Error: File '{image_path}' not found!")
        return
    
    # Create analyzer
    analyzer = OMRAnalyzer()
    
    # Analyze the OMR sheet
    answers = analyzer.analyze_omr(image_path)
    
    # Get formatted results
    results = analyzer.format_results(answers)
    
    # Generate output filename
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_file = f"results.txt"
    
    # Save results to text file
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"Image: {image_path}\n")
            f.write("=" * 60 + "\n\n")
            f.write(results)
        
        # Also save a simple answer key format
        if answers:
            answer_key_file = f"answer_key.txt"
            with open(answer_key_file, "w", encoding="utf-8") as f:
                f.write(f"Answer Key for {image_path}\n")
                f.write("=" * 40 + "\n\n")
                
                for q_num in sorted(answers.keys()):
                    f.write(f"{q_num}. {answers[q_num]}\n")
                
                f.write(f"\nTotal: {len(answers)} questions answered")
    
    except Exception as e:
        print(f"❌ Error saving results: {e}")

if __name__ == "__main__":
    main()