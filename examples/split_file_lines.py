#!/usr/bin/env python

# @file split_file_lines.py
#
# @date May, 2025
#
# @author Gino Francesco Bogo

import argparse


def split_file_lines(input_file, output_file1, output_file2, n):
    """
    Splits the input file into two files:
    - The first N lines are written to output_file1.
    - The remaining lines are written to output_file2.

    :param input_file: Path to the input text file
    :param output_file1: Path to the first output text file
    :param output_file2: Path to the second output text file
    :param n: Number of lines to move to the first output file
    """
    try:
        # Open the input file for reading
        with open(input_file, "r") as infile:
            # Read the first N lines
            first_n_lines = [next(infile) for _ in range(n)]
            # Read the remaining lines
            remaining_lines = infile.readlines()

        # Write the first N lines to the first output file
        with open(output_file1, "w") as outfile1:
            outfile1.writelines(first_n_lines)

        # Write the remaining lines to the second output file
        with open(output_file2, "w") as outfile2:
            outfile2.writelines(remaining_lines)

        print(
            f"Successfully split '{input_file}' into '{output_file1}' (first {n} lines) "
            f"and '{output_file2}' (remaining lines)."
        )
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' does not exist.")
    except StopIteration:
        print(
            f"Warning: The file '{input_file}' has fewer than {n} lines. "
            f"All available lines were written to '{output_file1}'."
        )
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(
        description="Split a file into two files: the first N lines and the remaining lines."
    )
    parser.add_argument(
        "-i", "--input", required=True, help="Path to the input text file"
    )
    parser.add_argument(
        "-o1", "--output1", required=True, help="Path to the first output text file"
    )
    parser.add_argument(
        "-o2", "--output2", required=True, help="Path to the second output text file"
    )
    parser.add_argument(
        "-n",
        "--lines",
        type=int,
        required=True,
        help="Number of lines to move to the first file",
    )

    args = parser.parse_args()

    # Call the function with parsed arguments
    split_file_lines(args.input, args.output1, args.output2, args.lines)
