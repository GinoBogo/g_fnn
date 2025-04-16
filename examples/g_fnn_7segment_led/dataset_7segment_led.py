#!/usr/bin/env python3

# @file dataset_7segment_led.py
#
# @date April, 2025
#
# @author Gino Francesco Bogo

import argparse
import random
import sys
import time

# Define the 7-segment LED patterns
#    ▄▄▄        A
#   █   █     F   B
#   ▀▄▄▄▀       G
#   █   █     E   C
#   ▀▄▄▄▀       D
#
patterns = {
    #   A  B  C  D  E  F  G
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

random.seed(time.time())

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Generate a dataset for 7-segment LED patterns."
)
parser.add_argument(
    "dataset_size", type=int, help="Size of the dataset (positive integer)."
)
parser.add_argument(
    "--error_max",
    type=float,
    default=0.0,
    help="Maximum value of error injection to add to 1.0 and 0.0 (default: 0.0).",
)
args = parser.parse_args()

# Validate dataset size
if args.dataset_size <= 0:
    print("Error: Dataset size must be a positive integer.")
    sys.exit(1)

# Validate error_max
if args.error_max < 0.0:
    print("Error: Maximum error value must be non-negative.")
    sys.exit(1)

# Open the output files
with open("fnn_dataset.set", "w") as inputs_file, open(
    "fnn_outputs.set", "w"
) as outputs_file:
    for _ in range(args.dataset_size):
        # Generate a random digit (0-9)
        digit = random.randint(0, 9)

        # Get the corresponding 7-segment LED pattern
        pattern = patterns[digit]

        # Convert the pattern to floats (0.0 = off, 1.0 = on) and inject errors
        inputs = [
            max(
                0.0,
                min(1.0, float(x) + random.uniform(-args.error_max, args.error_max)),
            )
            for x in pattern
        ]

        # Write the inputs to the file in formatted notation
        inputs_file.write(", ".join(f"{x:.4f}" for x in inputs) + "\n")

        # Generate the actual output
        output = [0] * 10
        output[digit] = 1

        # Write the actual output to the file
        outputs_file.write(", ".join(map(str, output)) + "\n")
