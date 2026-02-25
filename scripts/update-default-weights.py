#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.8"
# ///
"""Convert a trained weights.json to a bundled Lua default weights file.

Extracts neural network parameters needed for inference (weights, biases,
gammas, betas, running_means, running_vars) from a trained weights.json.
"""

import argparse
import json
import os
import sys


# Key ordering matching the existing nn_default_weights.lua format
NETWORK_KEY_ORDER = [
    "running_vars",
    "running_means",
    "weights",
    "betas",
    "gammas",
    "biases",
]


def infer_architecture(weights):
    """Infer the network architecture from weight matrix dimensions.

    The weight matrices encode the layer sizes:
    - Layer i has shape (input_size, output_size)
    - First layer's input_size is the network input dimension
    - Last layer's output_size is the network output dimension
    """
    if not weights:
        raise ValueError("No weight matrices found")

    arch = [len(weights[0])]  # input size from first layer's row count
    for layer in weights:
        arch.append(len(layer[0]))  # output size from each layer's column count
    return arch


def format_number(value):
    """Format a number for Lua output using repr() for full float precision."""
    return repr(value)


def format_array(values, indent_level):
    """Format a 1D array of numbers as a Lua table on a single line or multi-line."""
    indent = "  " * indent_level
    # For single-element arrays, put on one line: { value }
    if len(values) == 1:
        return f"{{ {format_number(values[0])} }}"

    # Multi-element arrays: one value per line
    lines = ["{"]
    inner_indent = "  " * (indent_level + 1)
    for val in values:
        lines.append(f"{inner_indent}{format_number(val)},")
    lines.append(f"{indent}}}")
    return "\n".join(lines)


def format_matrix(matrix, indent_level):
    """Format a 2D array (list of rows) as nested Lua tables."""
    indent = "  " * indent_level
    lines = ["{"]
    inner_indent = "  " * (indent_level + 1)
    for row in matrix:
        formatted_row = format_array(row, indent_level + 1)
        lines.append(f"{inner_indent}{formatted_row},")
    lines.append(f"{indent}}}")
    return "\n".join(lines)


def format_network_field(field_data, indent_level):
    """Format a network field (list of layer matrices) as nested Lua tables."""
    indent = "  " * indent_level
    lines = ["{"]
    inner_indent = "  " * (indent_level + 1)
    for layer_matrix in field_data:
        formatted_matrix = format_matrix(layer_matrix, indent_level + 1)
        lines.append(f"{inner_indent}{formatted_matrix},")
    lines.append(f"{indent}}}")
    return "\n".join(lines)


def generate_lua(version, network, architecture):
    """Generate the complete Lua file content."""
    arch_str = " -> ".join(str(s) for s in architecture)

    lines = []
    lines.append("--- Default neural network weights for neural-open")
    lines.append("--- Auto-generated from trained weights - DO NOT EDIT MANUALLY")
    lines.append("---")
    lines.append(
        "--- These weights are used as defaults when no user weights exist."
    )
    lines.append(
        "--- They represent a pre-trained network that provides good initial file ranking."
    )
    lines.append("---")
    lines.append(f"--- Network architecture: {arch_str}")
    version_labels = {
        "2.0-hinge": "v2.0 pairwise hinge loss",
    }
    lines.append(f"--- Training format: {version_labels.get(version, version)}")
    lines.append("")
    lines.append("return {")
    lines.append(f'  version = "{version}",')
    lines.append("  network = {")

    for key in NETWORK_KEY_ORDER:
        formatted = format_network_field(network[key], indent_level=2)
        lines.append(f'    ["{key}"] = {formatted},')

    lines.append("  },")
    lines.append("}")
    lines.append("")  # trailing newline

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Convert trained weights.json to bundled Lua default weights"
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="./weights.json",
        help="Path to weights.json (default: ./weights.json)",
    )
    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Read weights.json
    try:
        with open(args.input) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {args.input}: {e}", file=sys.stderr)
        sys.exit(1)

    # Navigate to nn.nn
    if "nn" not in data:
        print("Error: Missing 'nn' key in weights.json", file=sys.stderr)
        sys.exit(1)
    if "nn" not in data["nn"]:
        print("Error: Missing 'nn.nn' key in weights.json", file=sys.stderr)
        sys.exit(1)

    nn_data = data["nn"]["nn"]

    # Extract version
    if "version" not in nn_data:
        print("Error: Missing 'nn.nn.version' in weights.json", file=sys.stderr)
        sys.exit(1)
    version = nn_data["version"]

    # Extract network
    if "network" not in nn_data:
        print("Error: Missing 'nn.nn.network' in weights.json", file=sys.stderr)
        sys.exit(1)
    network = nn_data["network"]

    # Validate all required keys exist
    for key in NETWORK_KEY_ORDER:
        if key not in network:
            print(
                f"Error: Missing network key '{key}' in weights.json", file=sys.stderr
            )
            sys.exit(1)

    # Infer architecture from weight matrices
    architecture = infer_architecture(network["weights"])

    # Generate Lua output
    lua_content = generate_lua(version, network, architecture)

    # Determine output path (relative to script location -> repo root)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    output_path = os.path.join(
        repo_root, "lua", "neural-open", "algorithms", "nn_default_weights.lua"
    )

    # Write output
    with open(output_path, "w") as f:
        f.write(lua_content)

    print(f"Wrote default weights to {output_path}")
    print(f"  Architecture: {' -> '.join(str(s) for s in architecture)}")
    print(f"  Version: {version}")
    print(f"  Extracted: {', '.join(NETWORK_KEY_ORDER)}")


if __name__ == "__main__":
    main()
