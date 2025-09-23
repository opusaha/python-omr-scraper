#!/usr/bin/env python3
"""
OMR (Optical Mark Recognition) Analyzer
Analyzes OMR answer sheets and detects filled circles
Supports 4-column layout with 90 questions total
"""

import cv2
import numpy as np
import argparse
import sys
from typing import List, Tuple, Dict

class OMRAnalyzer:
    def __init__(self):
        self.image = None
        self.gray = None
        self.answers = {}
        
        # Configuration parameters
        self.min_circle_radius = 8
        self.max_circle_radius = 25
        self.filled_threshold = 0.6  # Threshold for determining if circle is filled
        
    def load_image(self, image_path: str) -> bool:
        """Load and preprocess the OMR image"""
        try:
            self.image = cv2.imread(image_path)
            if self.image is None:
                print(f"Error: Could not load image from {image_path}")
                return False
                
            # Convert to grayscale
            self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            return True
        except Exception as e:
            print(f"Error loading image: {e}")
            return False
    
    def detect_circles(self) -> List[Tuple[int, int, int]]:
        """Detect all circles in the image using HoughCircles"""
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(self.gray, (9, 9), 2)
        
        # Detect circles using HoughCircles
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=30,  # Minimum distance between circle centers
            param1=50,   # Higher threshold for edge detection
            param2=30,   # Accumulator threshold for center detection
            minRadius=self.min_circle_radius,
            maxRadius=self.max_circle_radius
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            return [(x, y, r) for x, y, r in circles]
        return []
    
    def is_circle_filled(self, x: int, y: int, radius: int) -> bool:
        """Check if a circle is filled by analyzing pixel intensity"""
        # Create a mask for the circle
        mask = np.zeros(self.gray.shape, dtype=np.uint8)
        cv2.circle(mask, (x, y), radius - 2, 255, -1)
        
        # Get pixels inside the circle
        circle_pixels = self.gray[mask == 255]
        
        if len(circle_pixels) == 0:
            return False
            
        # Calculate the percentage of dark pixels
        dark_pixels = np.sum(circle_pixels < 127)
        fill_ratio = dark_pixels / len(circle_pixels)
        
        return fill_ratio > self.filled_threshold
    
    def organize_circles_by_grid(self, circles: List[Tuple[int, int, int]]) -> Dict[int, Dict[int, List[Tuple[int, int, int]]]]:
        """Organize circles into a grid structure based on their positions"""
        if not circles:
            return {}
            
        # Sort circles by Y coordinate first (rows), then by X coordinate (columns)
        sorted_circles = sorted(circles, key=lambda c: (c[1], c[0]))
        
        # Group circles by rows (questions)
        rows = {}
        current_row = 0
        last_y = sorted_circles[0][1]
        row_tolerance = 20  # Pixels tolerance for same row
        
        for circle in sorted_circles:
            x, y, r = circle
            
            # Check if this circle is in a new row
            if abs(y - last_y) > row_tolerance:
                current_row += 1
                last_y = y
                
            if current_row not in rows:
                rows[current_row] = []
            rows[current_row].append(circle)
        
        # Organize each row into columns (options)
        grid = {}
        for row_idx, row_circles in rows.items():
            # Sort circles in this row by X coordinate
            row_circles.sort(key=lambda c: c[0])
            
            # Group into columns (every 4 circles should be options for one question)
            grid[row_idx] = {}
            for i, circle in enumerate(row_circles):
                col = i % 4  # Options 0, 1, 2, 3 (will be converted to 1, 2, 3, 4)
                if col not in grid[row_idx]:
                    grid[row_idx][col] = []
                grid[row_idx][col].append(circle)
        
        return grid
    
    def detect_question_layout(self, circles: List[Tuple[int, int, int]]) -> Dict[int, int]:
        """Detect which circles are filled and map to question numbers and answers"""
        filled_answers = {}
        
        # Get image dimensions
        height, width = self.gray.shape
        
        # Divide image into 4 vertical columns
        col_width = width // 4
        
        # Process each circle
        for x, y, r in circles:
            # Determine which column this circle is in (0-3)
            column = min(3, x // col_width)
            
            # Check if circle is filled
            if self.is_circle_filled(x, y, r):
                # Find other circles in the same row to determine question number and option
                row_circles = []
                for cx, cy, cr in circles:
                    if abs(cy - y) < 15:  # Same row tolerance
                        row_circles.append((cx, cy, cr))
                
                # Sort by x-coordinate to get option order
                row_circles.sort(key=lambda c: c[0])
                
                # Find this circle's position in the row
                option_index = -1
                for i, (cx, cy, cr) in enumerate(row_circles):
                    if cx == x and cy == y:
                        option_index = i % 4  # Option within the question (0-3)
                        break
                
                if option_index != -1:
                    # Calculate question number based on position
                    # Estimate row number
                    row_num = self.estimate_row_number(y, height)
                    # Calculate question number based on column and row
                    question_num = self.calculate_question_number(column, row_num)
                    
                    if 1 <= question_num <= 90:
                        filled_answers[question_num] = option_index + 1  # Convert to 1-4
        
        return filled_answers
    
    def estimate_row_number(self, y: int, height: int) -> int:
        """Estimate row number based on Y coordinate"""
        # Assuming roughly 23 questions per column
        # and questions are evenly distributed vertically
        return int((y / height) * 23)
    
    def calculate_question_number(self, column: int, row: int) -> int:
        """Calculate question number based on column and row"""
        # Column 0: Questions 1-23
        # Column 1: Questions 24-46  
        # Column 2: Questions 47-69
        # Column 3: Questions 70-90
        
        base_question = column * 23 + 1
        return base_question + row
    
    def flexible_layout_detection(self) -> Dict[int, int]:
        """Flexible method to detect any OMR layout with column-wise numbering"""
        circles = self.detect_circles()
        
        if not circles:
            print("No circles detected in the image")
            return {}
        
        print(f"Detected {len(circles)} circles")
        
        filled_answers = {}
        height, width = self.gray.shape
        
        # Step 1: Group circles by Y coordinate (rows)
        rows = {}
        row_tolerance = 25  # Tolerance for same row
        
        for x, y, r in circles:
            found_row = False
            for row_y in rows.keys():
                if abs(y - row_y) <= row_tolerance:
                    rows[row_y].append((x, y, r))
                    found_row = True
                    break
            
            if not found_row:
                rows[y] = [(x, y, r)]
        
        # Sort rows by Y coordinate (top to bottom)
        sorted_rows = sorted(rows.items(), key=lambda r: r[0])
        
        # Step 2: Analyze layout - detect number of columns
        sample_row_sizes = [len(row_circles) for _, row_circles in sorted_rows[:3]]
        avg_circles_per_row = sum(sample_row_sizes) / len(sample_row_sizes) if sample_row_sizes else 4
        
        print(f"Average circles per row: {avg_circles_per_row:.1f}")
        
        # Determine layout type
        if avg_circles_per_row <= 5:  # Single question per row (vertical layout)
            num_columns = 1
            questions_per_row = 1
        elif avg_circles_per_row <= 10:  # 2 questions per row
            num_columns = 2  
            questions_per_row = 2
        elif avg_circles_per_row <= 15:  # 3 questions per row
            num_columns = 3
            questions_per_row = 3
        else:  # 4+ questions per row
            num_columns = 4
            questions_per_row = 4
        
        print(f"Detected layout: {num_columns} columns, {questions_per_row} questions per row")
        
        # Step 3: Detect columns by X coordinates
        if num_columns > 1:
            # Find column boundaries
            all_x = [c[0] for row_circles in rows.values() for c in row_circles]
            all_x.sort()
            
            # Use K-means-like approach to find column centers
            column_centers = []
            if num_columns == 2:
                mid_x = width // 2
                column_centers = [width // 4, 3 * width // 4]
            elif num_columns == 3:
                column_centers = [width // 6, width // 2, 5 * width // 6]
            else:  # 4 columns
                column_centers = [width // 8, 3 * width // 8, 5 * width // 8, 7 * width // 8]
        
        # Step 4: Process each row and assign question numbers column-wise
        column_question_counts = [0] * num_columns  # Track questions per column
        
        for row_idx, (row_y, row_circles) in enumerate(sorted_rows):
            # Sort circles in this row by X coordinate (left to right)
            row_circles.sort(key=lambda c: c[0])
            
            if num_columns == 1:
                # Single column layout
                if len(row_circles) >= 4:  # Should have 4 option circles
                    option_circles = row_circles[-4:]  # Last 4 are options
                    question_num = row_idx + 1
                    
                    for opt_idx, (x, y, r) in enumerate(option_circles):
                        if self.is_circle_filled(x, y, r):
                            filled_answers[question_num] = opt_idx + 1
                            break
            
            else:
                # Multi-column layout
                # Group circles by columns
                col_groups = [[] for _ in range(num_columns)]
                
                for circle in row_circles:
                    x, y, r = circle
                    # Determine which column this circle belongs to
                    min_dist = float('inf')
                    best_col = 0
                    
                    for col_idx, center_x in enumerate(column_centers):
                        dist = abs(x - center_x)
                        if dist < min_dist:
                            min_dist = dist
                            best_col = col_idx
                    
                    col_groups[best_col].append(circle)
                
                # Process each column in this row
                for col_idx, col_circles in enumerate(col_groups):
                    if len(col_circles) >= 4:  # Should have 4 option circles
                        col_circles.sort(key=lambda c: c[0])  # Sort by X within column
                        option_circles = col_circles[-4:]  # Last 4 are options
                        
                        # Calculate question number based on column-wise numbering
                        # Different logic for different layouts:
                        # Hope Wheeler (2 col): Row-based (Q1-Q10 left, Q11-Q20 right)
                        # Nexes (4 col): Column-based (Q1-Q23 col1, Q24-Q46 col2, etc.)
                        
                        current_row = row_idx
                        if num_columns == 2:
                            # Hope Wheeler style: row-based numbering
                            if col_idx == 0:  # Left column
                                question_num = current_row + 1
                            else:  # Right column  
                                question_num = current_row + 11
                        else:
                            # Nexes style: column-based numbering
                            # Each column gets 23 questions except last column
                            questions_per_col = 23
                            base_question = col_idx * questions_per_col + 1
                            question_num = base_question + current_row
                        
                        # Check which option is filled
                        filled_option = None
                        for opt_idx, (x, y, r) in enumerate(option_circles):
                            if self.is_circle_filled(x, y, r):
                                filled_option = opt_idx + 1
                                break
                        
                        if filled_option:
                            filled_answers[question_num] = filled_option
                            column_question_counts[col_idx] += 1  # Increment question count for this column
        
        print(f"Questions per column: {column_question_counts}")
        return filled_answers
    
    def advanced_circle_detection(self) -> Dict[int, int]:
        """Advanced method with flexible layout detection"""
        return self.flexible_layout_detection()
    
    def analyze_omr(self, image_path: str) -> Dict[int, int]:
        """Main method to analyze OMR sheet"""
        if not self.load_image(image_path):
            return {}
        
        print(f"Analyzing OMR sheet: {image_path}")
        
        # Use advanced detection method
        answers = self.advanced_circle_detection()
        
        return answers
    
    def format_results(self, answers: Dict[int, int]) -> str:
        """Format results with flexible column-wise display"""
        if not answers:
            return "কোন উত্তর পাওয়া যায়নি।"
        
        result = "OMR Analysis Results:\n"
        result += "=" * 50 + "\n\n"
        
        if not answers:
            return result + "কোন উত্তর পাওয়া যায়নি।"
        
        # Get max question number to determine layout
        max_question = max(answers.keys())
        min_question = min(answers.keys())
        total_questions = max_question
        
        # Detect column structure from answer distribution
        question_nums = sorted(answers.keys())
        
        # Group questions into columns based on numbering pattern
        columns = []
        current_column = []
        
        # Simple approach: group consecutive numbers, max 25 per column
        for i, q_num in enumerate(question_nums):
            if len(current_column) >= 25:  # Max 25 per column
                columns.append(current_column)
                current_column = [q_num]
            elif i > 0 and q_num - question_nums[i-1] > 5:  # Gap indicates new column
                columns.append(current_column)
                current_column = [q_num]
            else:
                current_column.append(q_num)
        
        if current_column:
            columns.append(current_column)
        
        # If we don't have clear column separation, create based on count
        if len(columns) == 1 and len(question_nums) > 25:
            # Split into multiple columns
            cols_needed = min(4, (len(question_nums) + 24) // 25)  # Max 4 columns
            items_per_col = len(question_nums) // cols_needed
            
            columns = []
            for i in range(cols_needed):
                start_idx = i * items_per_col
                end_idx = start_idx + items_per_col if i < cols_needed - 1 else len(question_nums)
                columns.append(question_nums[start_idx:end_idx])
        
        # Display results column by column
        for col_idx, col_questions in enumerate(columns):
            if col_questions:
                min_q = min(col_questions)
                max_q = max(col_questions)
                
                result += f"\nকলাম {col_idx + 1} ({min_q}-{max_q}):  [{len(col_questions)}টি প্রশ্ন]\n"
                
                # Show all questions in range, indicating missing ones
                for q in range(min_q, max_q + 1):
                    if q in answers:
                        result += f"প্রশ্ন {q}: {answers[q]} নম্বর অপশন\n"
                    else:
                        result += f"প্রশ্ন {q}: উত্তর পাওয়া যায়নি\n"
        
        # Summary
        result += f"\n\n**সারসংক্ষেপ:**\n"
        result += f"- **সর্বোচ্চ প্রশ্ন নম্বর:** {max_question}\n"
        result += f"- **মোট কলাম:** {len(columns)}টি\n"
        result += f"- **উত্তর পাওয়া গেছে:** {len(answers)}টি\n"
        result += f"- **অনুপস্থিত উত্তর:** {max_question - len(answers)}টি\n"
        
        # Column distribution
        result += f"- **কলাম অনুযায়ী বন্টন:** "
        for i, col_questions in enumerate(columns):
            result += f"কলাম {i+1}: {len(col_questions)}টি"
            if i < len(columns) - 1:
                result += ", "
        result += "\n"
        
        return result

def main():
    parser = argparse.ArgumentParser(description='OMR Sheet Analyzer')
    parser.add_argument('image_path', help='Path to the OMR sheet image')
    parser.add_argument('--output', '-o', help='Output file path (optional)')
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Create analyzer instance
    analyzer = OMRAnalyzer()
    
    # Analyze the OMR sheet
    answers = analyzer.analyze_omr(args.image_path)
    
    # Format and display results
    results = analyzer.format_results(answers)
    
    print(results)
    
    # Save to file if output path specified
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(results)
            print(f"\nResults saved to: {args.output}")
        except Exception as e:
            print(f"Error saving results: {e}")

if __name__ == "__main__":
    main()
