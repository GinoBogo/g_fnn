#!/usr/bin/env python

# @file convert_7segment_to_digits.py
#
# @date May, 2025
#
# @author Gino Francesco Bogo


import csv
import argparse

# Define the 7-segment patterns for each digit (0-9)
SEVEN_SEGMENT_PATTERNS = {
    0: [1, 1, 1, 1, 1, 1, 0],  # 0
    1: [0, 1, 1, 0, 0, 0, 0],  # 1
    2: [1, 1, 0, 1, 1, 0, 1],  # 2
    3: [1, 1, 1, 1, 0, 0, 1],  # 3
    4: [0, 1, 1, 0, 0, 1, 1],  # 4
    5: [1, 0, 1, 1, 0, 1, 1],  # 5
    6: [1, 0, 1, 1, 1, 1, 1],  # 6
    7: [1, 1, 1, 0, 0, 0, 0],  # 7
    8: [1, 1, 1, 1, 1, 1, 1],  # 8
    9: [1, 1, 1, 1, 0, 1, 1],  # 9
}


def pattern_to_digit(pattern):
    """Convert a 7-segment pattern to its corresponding digit."""
    for digit, template in SEVEN_SEGMENT_PATTERNS.items():
        if pattern == template:
            return digit
    return None  # Return None if no match is found


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Convert 7-segment LED patterns to digit representations"
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to the input CSV file with 7-segment patterns",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Path to the output CSV file with digit representations",
    )
    args = parser.parse_args()

    input_file = args.input
    output_file = args.output

    # Read input file
    with open(input_file, "r") as f:
        reader = csv.reader(f)
        patterns = list(reader)

    # Convert patterns to digits and create output
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)

        for pattern in patterns:
            # Convert string pattern to integers
            int_pattern = list(map(int, pattern))

            # Get the corresponding digit
            digit = pattern_to_digit(int_pattern)

            # Create output row with 10 columns
            if digit is not None:
                output_row = [0] * 10
                output_row[digit] = 1
                writer.writerow(output_row)
            else:
                print(f"Warning: Pattern {pattern} doesn't match any digit")


if __name__ == "__main__":
    main()
