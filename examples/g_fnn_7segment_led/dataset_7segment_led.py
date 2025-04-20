#!/usr/bin/env python3

"""
@file dataset_7segment_led.py

@date April, 2025

@author Gino Francesco Bogo

@license MIT License

@version 1.0.0
"""

import random
import time
from pathlib import Path


def get_input(prompt, default=None, validate=None):
    """Get user input with validation and default value."""
    while True:
        try:
            value = input(prompt)
            if not value and default is not None:
                return default
            value = validate(value) if validate else value
            return value
        except (ValueError, TypeError) as e:
            print(f"Error: {e}")


def validate_positive_int(value):
    """Validate that a value is a positive integer."""
    value = int(value)
    if value <= 0:
        raise ValueError("Value must be a positive integer")
    return value


def validate_float_range(value, min_val=0.0, max_val=0.5):
    """Validate that a float is within a specified range."""
    value = float(value)
    if not (min_val <= value < max_val):
        raise ValueError(f"Value must be in the range [{min_val}, {max_val})")
    return value


def validate_filename(value, default=None, counter=0):
    """Validate and format a filename. Ensures unique filenames by appending a
    counter if needed."""
    if not value:
        if default:
            return default
        return "fnn_dataset.set"

    # Add .set extension if not present
    if not value.endswith(".set"):
        value += ".set"

    # If this is the first call and we have a default, check if the value exists
    if counter == 0 and default:
        default_path = Path(default)
        if default_path.exists():
            # If default exists, use the provided value
            return value

    # Check if file exists
    file_path = Path(value)
    if file_path.exists():
        # If file exists, add a counter to make it unique
        base_name = file_path.stem
        if counter > 0:
            base_name = base_name.rsplit("_", 1)[
                0
            ]  # Remove previous counter if present
        new_name = f"{base_name}_{counter}.set"
        return validate_filename(new_name, default, counter + 1)

    return value


def generate_dataset(size, error_max):
    """
    Generate a dataset of 7-segment LED patterns.
    """
    inputs = []
    outputs = []

    # Define the 7-segment LED patterns
    #    ▄▄▄        A
    #   █   █     F   B
    #   ▀▄▄▄▀       G
    #   █   █     E   C
    #   ▀▄▄▄▀       D
    #

    # 7-segment LED patterns for digits 0-9
    patterns = {
        0: [1, 1, 1, 1, 1, 1, 0],
        1: [0, 1, 1, 0, 0, 0, 0],
        2: [1, 1, 0, 1, 1, 0, 1],
        3: [1, 1, 1, 1, 0, 0, 1],
        4: [0, 1, 1, 0, 0, 1, 1],
        5: [1, 0, 1, 1, 0, 1, 1],
        6: [1, 0, 1, 1, 1, 1, 1],
        7: [1, 1, 1, 0, 0, 0, 0],
        8: [1, 1, 1, 1, 1, 1, 1],
        9: [1, 1, 1, 1, 0, 1, 1],
    }

    error_max = float(error_max)

    for _ in range(size):
        digit = random.randint(0, 9)
        pattern = patterns[digit]

        # Add random noise to the pattern
        noisy_pattern = [
            max(0.0, min(1.0, float(x) + random.uniform(-error_max, error_max)))
            for x in pattern
        ]

        inputs.append(noisy_pattern)

        # Create one-hot encoded output
        output = [0] * 10
        output[digit] = 1
        outputs.append(output)

    return inputs, outputs


def save_dataset(inputs, outputs, dataset_file, outputs_file):
    """Save the generated dataset to files."""
    # Create output directory if it doesn't exist
    output_dir = Path(dataset_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save inputs
    with open(dataset_file, "w") as f:
        for pattern in inputs:
            f.write(", ".join(f"{x:.4f}" for x in pattern) + "\n")

    # Save outputs
    with open(outputs_file, "w") as f:
        for pattern in outputs:
            f.write(", ".join(map(str, pattern)) + "\n")


def main():
    # Display presentation
    print("\n")
    print(" =========================================")
    print("  7-Segment LED Pattern Dataset Generator ")
    print(" =========================================")
    print("\n")
    print(" This script generates a dataset for training a neural network to recognize")
    print(" 7-segment LED patterns. Each pattern consists of 7 segments (A-G) that can")
    print(" be either on (1.0) or off (0.0). An error injection is also possible.")
    print("\n")
    print("     ▄▄▄        A")
    print("    █   █     F   B")
    print("    ▀▄▄▄▀       G")
    print("    █   █     E   C")
    print("    ▀▄▄▄▀       D")
    print("\n")
    print(" Example patterns:")
    print(" - Digit 0: [1, 1, 1, 1, 1, 1, 0] (all segments on except G)")
    print(" - Digit 1: [0, 1, 1, 0, 0, 0, 0] (segments B and C on)\n")

    # Seed random number generator
    random.seed(time.time())

    # Get user inputs
    print(" Parameters:")
    print(" ----------")
    size = get_input(
        " Enter dataset size (positive integer): ", validate=validate_positive_int
    )

    error_max = get_input(
        " Enter maximum error value (0.0 <= value < 0.5) (default: 0.0): ",
        default="0.0",
        validate=lambda x: validate_float_range(x, 0.0, 0.5),
    )

    dataset_file = get_input(
        " Enter dataset filename (default: fnn_dataset.set): ",
        validate=lambda x: validate_filename(x, "fnn_dataset.set"),
    )

    outputs_file = get_input(
        " Enter outputs filename (default: fnn_outputs.set): ",
        validate=lambda x: validate_filename(x, "fnn_outputs.set"),
    )

    print("\n")
    print(" Generating files...")

    # Generate and save the files
    inputs, outputs = generate_dataset(size, error_max)
    save_dataset(inputs, outputs, dataset_file, outputs_file)

    print("\n")
    print(f" Files generated successfully!")
    print(f" Dataset size: {size}")
    print(f" Dataset file: {dataset_file}")
    print(f" Outputs file: {outputs_file}")


if __name__ == "__main__":
    main()
