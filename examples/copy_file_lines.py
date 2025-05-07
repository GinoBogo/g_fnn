#!/usr/bin/env python

# @file copy_file_lines.py
#
# @date May, 2025
#
# @author Gino Francesco Bogo

import argparse


def copy_n_lines(input_file, output_file, n):
    """
    Copies the first N lines from the input file to the output file.

    :param input_file: Path to the input text file
    :param output_file: Path to the output text file
    :param n: Number of lines to copy
    """
    try:
        # Open the input file for reading
        with open(input_file, "r") as infile:
            # Read the first N lines
            lines = [next(infile) for _ in range(n)]

        # Open the output file for writing
        with open(output_file, "w") as outfile:
            # Write the lines to the output file
            outfile.writelines(lines)

        print(f"Successfully copied {n} lines from '{input_file}' to '{output_file}'.")
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' does not exist.")
    except StopIteration:
        print(
            f"Warning: The file '{input_file}' has fewer than {n} lines. Copied all available lines."
        )
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(
        description="Copy the first N lines from one file to another."
    )
    parser.add_argument(
        "-i", "--input", required=True, help="Path to the input text file"
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Path to the output text file"
    )
    parser.add_argument(
        "-n", "--lines", type=int, required=True, help="Number of lines to copy"
    )

    args = parser.parse_args()

    # Call the function with parsed arguments
    copy_n_lines(args.input, args.output, args.lines)
