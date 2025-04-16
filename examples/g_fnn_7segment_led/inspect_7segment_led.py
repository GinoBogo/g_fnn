#!/usr/bin/env python3

# @file inspect_7segment_led.py
#
# @date April, 2025
#
# @author Gino Francesco Bogo


import sys
import os


def process_line(line):
    """Process a single line of 10 floats and return binary representation."""
    # Split the line by commas and convert to floats
    values = [float(x) for x in line.strip().split(",")]

    if len(values) != 10:
        raise ValueError(f"Expected 10 values, got {len(values)}")

    # Find the maximum value
    max_value = max(values)

    # Create binary representation
    binary_output = [1 if x == max_value else 0 for x in values]

    # Convert to string with commas and spaces
    return ", ".join(map(str, binary_output))


def main():
    if len(sys.argv) != 3:
        print(f"Usage: python {os.path.basename(__file__)} <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist")
        sys.exit(1)

    try:
        with open(input_file, "r") as infile, open(output_file, "a") as outfile:
            for line in infile:
                if line.strip():  # Skip empty lines
                    processed_line = process_line(line)
                    outfile.write(processed_line + "\n")

        print(f"Processed {input_file} and appended results to {output_file}")

    except Exception as e:
        print(f"Error processing files: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
