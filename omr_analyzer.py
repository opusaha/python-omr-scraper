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
        """Detect all circles in the image using improved HoughCircles"""
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(self.gray, (5, 5), 1.5)
        
        # Try multiple parameter sets for better detection
        all_circles = []
        
        # Parameter set 1: Standard detection
        circles1 = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=25,  # Reduced minimum distance
            param1=50,
            param2=25,   # Lower accumulator threshold
            minRadius=self.min_circle_radius,
            maxRadius=self.max_circle_radius
        )
        
        if circles1 is not None:
            circles1 = np.round(circles1[0, :]).astype("int")
            all_circles.extend([(x, y, r) for x, y, r in circles1])
        
        # Parameter set 2: More sensitive detection
        circles2 = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=40,   # Lower edge threshold
            param2=20,   # Even lower accumulator threshold
            minRadius=self.min_circle_radius,
            maxRadius=self.max_circle_radius
        )
        
        if circles2 is not None:
            circles2 = np.round(circles2[0, :]).astype("int")
            all_circles.extend([(x, y, r) for x, y, r in circles2])
        
        # Remove duplicates (circles that are very close to each other)
        unique_circles = self._remove_duplicate_circles(all_circles)
        
        print(f"HoughCircles detected {len(all_circles)} total, {len(unique_circles)} unique circles")
        return unique_circles
    
    def _remove_duplicate_circles(self, circles: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
        """Remove duplicate circles that are too close to each other"""
        if not circles:
            return []
        
        unique_circles = []
        min_distance = 15  # Minimum distance between circle centers
        
        for x, y, r in circles:
            is_duplicate = False
            for ux, uy, ur in unique_circles:
                distance = np.sqrt((x - ux)**2 + (y - uy)**2)
                if distance < min_distance:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_circles.append((x, y, r))
        
        return unique_circles
    
    def is_circle_filled(self, x: int, y: int, radius: int) -> bool:
        """Improved method to check if a circle is filled by analyzing pixel intensity"""
        # Ensure coordinates are within image bounds
        height, width = self.gray.shape
        if x < radius or y < radius or x >= width - radius or y >= height - radius:
            return False
        
        # Create a mask for the circle with slightly smaller radius to avoid edge effects
        mask = np.zeros(self.gray.shape, dtype=np.uint8)
        inner_radius = max(1, radius - 3)  # Use smaller radius to focus on center
        cv2.circle(mask, (x, y), inner_radius, 255, -1)
        
        # Get pixels inside the circle
        circle_pixels = self.gray[mask == 255]
        
        if len(circle_pixels) == 0:
            return False
        
        # Calculate statistics
        mean_intensity = np.mean(circle_pixels)
        median_intensity = np.median(circle_pixels)
        
        # Use multiple criteria for better accuracy
        # Criterion 1: Dark pixel percentage
        dark_pixels = np.sum(circle_pixels < 120)  # Slightly higher threshold
        fill_ratio = dark_pixels / len(circle_pixels)
        
        # Criterion 2: Mean intensity (filled circles should be darker)
        mean_threshold = mean_intensity < 100
        
        # Criterion 3: Median intensity (more robust to outliers)
        median_threshold = median_intensity < 110
        
        # Circle is considered filled if it meets multiple criteria
        is_filled = (fill_ratio > 0.5) and (mean_threshold or median_threshold)
        
        # Debug output for troubleshooting
        if is_filled:
            print(f"Filled circle at ({x}, {y}): fill_ratio={fill_ratio:.2f}, mean={mean_intensity:.1f}, median={median_intensity:.1f}")
        
        return is_filled
    
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
        """Improved method with accurate column-wise sequential numbering for 2-4 columns"""
        circles = self.detect_circles()
        
        if not circles:
            print("No circles detected in the image")
            return {}
        
        print(f"Detected {len(circles)} total circles")
        
        filled_answers = {}
        height, width = self.gray.shape
        
        # Step 1: Detect columns by analyzing X-coordinate distribution
        all_x_coords = [c[0] for c in circles]
        columns = self._detect_columns_by_clustering(all_x_coords, width)
        num_columns = len(columns)
        
        print(f"Detected {num_columns} columns with boundaries: {columns}")
        
        # Step 2: Group circles by rows (Y coordinate)
        rows = self._group_circles_by_rows(circles)
        sorted_rows = sorted(rows.items(), key=lambda r: r[0])
        
        print(f"Detected {len(sorted_rows)} rows")
        
        # Step 3: Process columns one by one (column-wise numbering)
        column_question_counts = [0] * num_columns
        question_counter = 1  # Sequential question numbering
        
        # Process each column from left to right
        for col_idx in range(num_columns):
            print(f"\nProcessing Column {col_idx + 1}:")
            
            # For each row, check if this column has circles
            for row_idx, (row_y, row_circles) in enumerate(sorted_rows):
                # Sort circles in this row by X coordinate
                row_circles.sort(key=lambda c: c[0])
                
                # Group circles by columns
                col_groups = self._assign_circles_to_columns(row_circles, columns)
                
                # Get circles for current column
                col_circles = col_groups.get(col_idx, [])
                
                if len(col_circles) >= 4:  # Should have at least 4 option circles
                    col_circles.sort(key=lambda c: c[0])  # Sort by X within column
                    
                    # Take the last 4 circles as option circles (in case there are extra circles)
                    option_circles = col_circles[-4:]
                    
                    # Current question number
                    current_question = question_counter
                    question_counter += 1
                    
                    print(f"  Row {row_idx + 1}: Question {current_question}")
                    
                    # Check which option is filled
                    filled_option = None
                    for opt_idx, (x, y, r) in enumerate(option_circles):
                        if self.is_circle_filled(x, y, r):
                            filled_option = opt_idx + 1
                            print(f"    Question {current_question}: Option {filled_option} is filled at ({x}, {y})")
                            break
                    
                    if filled_option:
                        filled_answers[current_question] = filled_option
                        column_question_counts[col_idx] += 1
        
        print(f"Questions per column: {column_question_counts}")
        print(f"Total questions detected: {sum(column_question_counts)}")
        return filled_answers
    
    def _detect_columns_by_clustering(self, x_coords: List[int], width: int) -> List[Tuple[int, int]]:
        """Detect column boundaries by clustering X coordinates"""
        if not x_coords:
            return []
        
        # Sort X coordinates
        sorted_x = sorted(set(x_coords))
        
        # Find gaps to determine column boundaries
        gaps = []
        for i in range(1, len(sorted_x)):
            gap = sorted_x[i] - sorted_x[i-1]
            if gap > 50:  # Significant gap indicates column separation
                gaps.append((sorted_x[i-1], sorted_x[i], gap))
        
        # Determine number of columns based on gaps
        if len(gaps) == 0:
            # Single column
            return [(0, width)]
        elif len(gaps) == 1:
            # Two columns
            mid_point = (gaps[0][0] + gaps[0][1]) // 2
            return [(0, mid_point), (mid_point, width)]
        elif len(gaps) == 2:
            # Three columns
            mid1 = (gaps[0][0] + gaps[0][1]) // 2
            mid2 = (gaps[1][0] + gaps[1][1]) // 2
            return [(0, mid1), (mid1, mid2), (mid2, width)]
        else:
            # Four columns (take largest 3 gaps)
            gaps.sort(key=lambda g: g[2], reverse=True)
            boundaries = []
            for gap in gaps[:3]:
                boundaries.append((gap[0] + gap[1]) // 2)
            boundaries.sort()
            
            columns = [(0, boundaries[0])]
            for i in range(1, len(boundaries)):
                columns.append((boundaries[i-1], boundaries[i]))
            columns.append((boundaries[-1], width))
            return columns
    
    def _group_circles_by_rows(self, circles: List[Tuple[int, int, int]]) -> Dict[int, List[Tuple[int, int, int]]]:
        """Group circles by rows using Y coordinate with dynamic tolerance"""
        rows = {}
        row_tolerance = 20  # Base tolerance
        
        for x, y, r in circles:
            found_row = False
            
            # Try to find existing row within tolerance
            for row_y in list(rows.keys()):
                if abs(y - row_y) <= row_tolerance:
                    rows[row_y].append((x, y, r))
                    found_row = True
                    break
            
            if not found_row:
                rows[y] = [(x, y, r)]
        
        return rows
    
    def _assign_circles_to_columns(self, row_circles: List[Tuple[int, int, int]], columns: List[Tuple[int, int]]) -> Dict[int, List[Tuple[int, int, int]]]:
        """Assign circles in a row to their respective columns"""
        col_groups = {}
        
        for circle in row_circles:
            x, y, r = circle
            
            # Find which column this circle belongs to
            for col_idx, (col_start, col_end) in enumerate(columns):
                if col_start <= x < col_end:
                    if col_idx not in col_groups:
                        col_groups[col_idx] = []
                    col_groups[col_idx].append(circle)
                    break
        
        return col_groups
    
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
    
    def mark_unmatched_answers(self, image_path: str, unmatched_data: Dict, correct_answers: Dict) -> str:
        """Mark correct answers for unmatched questions with green circles and save the modified image"""
        if not self.load_image(image_path):
            return ""
        
        # Detect all circles first
        circles = self.detect_circles()
        if not circles:
            print("No circles detected for marking")
            return ""
        
        # Create a copy of the image for marking
        marked_image = self.image.copy()
        
        # Get image dimensions
        height, width = self.gray.shape
        
        # Use the same logic as flexible_layout_detection
        all_x_coords = [c[0] for c in circles]
        columns = self._detect_columns_by_clustering(all_x_coords, width)
        num_columns = len(columns)
        
        print(f"Detected {num_columns} columns for marking: {columns}")
        
        # Group circles by rows
        rows = self._group_circles_by_rows(circles)
        sorted_rows = sorted(rows.items(), key=lambda r: r[0])
        
        print(f"Detected {len(sorted_rows)} rows for marking")
        
        # Mark correct answers for unmatched questions with green circles
        for question_num, answer_info in unmatched_data.items():
            correct_answer = answer_info['correct']
            
            # Convert Bengali letter to option number (1-4)
            option_number = self._convert_bangla_to_option_number(correct_answer)
            if option_number is None:
                continue
            
            print(f"Marking question {question_num}: correct answer '{correct_answer}' (option {option_number})")
                
            # Find circles for this specific question using the same logic as analysis
            question_circles = self._find_circles_for_question_dynamic(
                circles, question_num, sorted_rows, columns
            )
            
            # Sort circles by X coordinate to get option order (ক, খ, গ, ঘ)
            question_circles.sort(key=lambda c: c[0])
            
            print(f"Found {len(question_circles)} circles for question {question_num}")
            
            # Mark the correct option with green circle
            if len(question_circles) >= option_number:
                x, y, r = question_circles[option_number - 1]  # -1 because list is 0-indexed
                print(f"Marking option {option_number} at position ({x}, {y}) with radius {r}")
                
                # Draw a thick green circle around the correct option
                cv2.circle(marked_image, (x, y), r + 5, (0, 255, 0), 4)  # Green color in BGR
                # Also add a green fill with transparency effect
                overlay = marked_image.copy()
                cv2.circle(overlay, (x, y), r - 2, (0, 255, 0), -1)
                marked_image = cv2.addWeighted(marked_image, 0.7, overlay, 0.3, 0)
            else:
                print(f"Warning: Not enough circles found for question {question_num} (found {len(question_circles)}, need {option_number})")
        
        # Save the marked image in the project root directory
        import os
        filename = os.path.basename(image_path)
        name, ext = os.path.splitext(filename)
        
        # Create marked images directory in project root
        project_root = os.getcwd()  # Get current working directory (project root)
        marked_dir = os.path.join(project_root, 'marked_images')
        
        # Ensure directory exists
        if not os.path.exists(marked_dir):
            os.makedirs(marked_dir, exist_ok=True)
            print(f"Created marked_images directory at: {marked_dir}")
        
        output_path = os.path.join(marked_dir, f"{name}_marked{ext}")
        
        # Save the image and verify it was saved
        success = cv2.imwrite(output_path, marked_image)
        if success:
            print(f"✅ Marked image successfully saved to: {output_path}")
            # Verify file exists
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                print(f"✅ File verified: {output_path} ({file_size} bytes)")
            else:
                print(f"❌ File not found after saving: {output_path}")
                return ""
        else:
            print(f"❌ Failed to save marked image to: {output_path}")
            return ""
        
        return output_path
    
    def _convert_bangla_to_option_number(self, bangla_letter: str) -> int:
        """Convert Bengali letter to option number (1-4)"""
        conversion_map = {
            'ক': 1,  # ka
            'খ': 2,  # kha
            'গ': 3,  # ga
            'ঘ': 4   # gha
        }
        return conversion_map.get(bangla_letter)
    
    def _find_circles_for_question_dynamic(self, circles: List[Tuple[int, int, int]], question_num: int, sorted_rows: List, columns: List[Tuple[int, int]]) -> List[Tuple[int, int, int]]:
        """Find circles for a specific question using the new column-wise sequential numbering logic"""
        
        # Use the same column-wise logic as flexible_layout_detection
        question_counter = 1
        num_columns = len(columns)
        
        # Process each column from left to right (same as analysis)
        for col_idx in range(num_columns):
            # For each row, check if this column has circles
            for row_idx, (row_y, row_circles) in enumerate(sorted_rows):
                # Sort circles in this row by X coordinate
                row_circles.sort(key=lambda c: c[0])
                
                # Group circles by columns
                col_groups = self._assign_circles_to_columns(row_circles, columns)
                
                # Get circles for current column
                col_circles = col_groups.get(col_idx, [])
                
                if len(col_circles) >= 4:  # Should have at least 4 option circles
                    col_circles.sort(key=lambda c: c[0])  # Sort by X within column
                    
                    # Check if this is our target question
                    if question_counter == question_num:
                        return col_circles[-4:]  # Return last 4 circles (option circles)
                    
                    question_counter += 1
        
        return []  # Question not found
    
    def _find_circles_for_question(self, circles: List[Tuple[int, int, int]], question_num: int, height: int, width: int) -> List[Tuple[int, int, int]]:
        """Find all option circles for a specific question number"""
        # Determine which column this question belongs to based on numbering
        if question_num <= 23:
            column = 0
        elif question_num <= 46:
            column = 1
        elif question_num <= 69:
            column = 2
        else:
            column = 3
        
        # Calculate expected row for this question
        if column == 0:
            row_in_column = question_num - 1
        elif column == 1:
            row_in_column = question_num - 24
        elif column == 2:
            row_in_column = question_num - 47
        else:
            row_in_column = question_num - 70
        
        # Estimate Y coordinate for this row
        expected_y = int((row_in_column / 23) * height) + 50  # Add some offset from top
        
        # Find circles near this expected position
        question_circles = []
        y_tolerance = 30  # Tolerance for Y coordinate matching
        
        for x, y, r in circles:
            # Check if this circle is in the right vertical position
            if abs(y - expected_y) < y_tolerance:
                # Check if it's in the right column area
                col_width = width // 4
                col_start = column * col_width
                col_end = (column + 1) * col_width
                
                if col_start <= x <= col_end:
                    question_circles.append((x, y, r))
        
        return question_circles

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
