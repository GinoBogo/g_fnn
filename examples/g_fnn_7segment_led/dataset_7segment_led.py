# -----------------------------------------------------------------------------
# @file dataset_7segment_led.py
#
# @date April, 2025
#
# @author Gino Francesco Bogo
# -----------------------------------------------------------------------------

import random
import time
import sys

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

# Check if the required arguments are provided
if len(sys.argv) != 2:
    print(f"Usage: python {sys.argv[0]} <dataset_size>")
    sys.exit(1)

try:
    # Get the dataset size from the arguments
    dataset_size = int(sys.argv[1])
    if dataset_size <= 0:
        raise ValueError("Dataset size must be a positive integer.")
except ValueError as e:
    print(f"Error: {e}")
    print(f"Usage: python {sys.argv[0]} <dataset_size>")
    sys.exit(1)

# Open the output files
with open("network_inputs.txt", "w") as inputs_file, open(
    "actual_outputs.txt", "w"
) as outputs_file:
    for _ in range(dataset_size):
        # Generate a random digit (0-9)
        digit = random.randint(0, 9)

        # Get the corresponding 7-segment LED pattern
        pattern = patterns[digit]

        # Convert the pattern to floats (0.0 = off, 1.0 = on)
        inputs = [float(x) for x in pattern]

        # Write the inputs to the file in scientific notation
        inputs_file.write(" ".join(f"{x:.4f}" for x in inputs) + "\n")

        # Generate the actual output
        output = [0] * 10
        output[digit] = 1

        # Write the actual output to the file
        outputs_file.write(" ".join(map(str, output)) + "\n")
