#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.8"
# ///
"""Convert trained weight files to bundled Lua default weight files.

Reads picker weight files (files.json, just_recipes.json) from a directory
and extracts neural network parameters needed for inference (weights, biases,
gammas, betas, running_means, running_vars) into Lua files.
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

# Picker source files and their corresponding Lua output filenames
PICKERS = [
    {
        "source": "files.json",
        "output": "nn_default_weights.lua",
        "description": "file ranking",
    },
    {
        "source": "just_recipes.json",
        "output": "nn_item_default_weights.lua",
        "description": "item ranking",
    },
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


def generate_lua(version, network, architecture, description):
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
        f"--- They represent a pre-trained network that provides good initial {description}."
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


def extract_nn_data(data, source_file):
    """Extract NN data from a weight file, handling both flat and legacy formats.

    Current format: { "nn": { "version": ..., "network": ..., ... } }
    Legacy format:  { "nn": { "nn": { "version": ..., "network": ..., ... } } }
    """
    if "nn" not in data:
        print(f"Error: Missing 'nn' key in {source_file}", file=sys.stderr)
        return None

    nn_data = data["nn"]

    # Auto-migrate legacy double-nested format
    if "nn" in nn_data and isinstance(nn_data["nn"], dict) and "network" in nn_data["nn"]:
        nn_data = nn_data["nn"]

    if "version" not in nn_data:
        print(f"Error: Missing 'version' in {source_file}", file=sys.stderr)
        return None

    if "network" not in nn_data:
        print(f"Error: Missing 'network' in {source_file}", file=sys.stderr)
        return None

    network = nn_data["network"]
    for key in NETWORK_KEY_ORDER:
        if key not in network:
            print(
                f"Error: Missing network key '{key}' in {source_file}", file=sys.stderr
            )
            return None

    return nn_data


def process_picker(input_dir, picker, output_dir):
    """Process a single picker weight file and generate its Lua output."""
    source_path = os.path.join(input_dir, picker["source"])

    if not os.path.exists(source_path):
        print(f"  Skipping {picker['source']} (not found)")
        return False

    # Read weight file
    try:
        with open(source_path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {source_path}: {e}", file=sys.stderr)
        return False

    nn_data = extract_nn_data(data, picker["source"])
    if nn_data is None:
        return False

    version = nn_data["version"]
    network = nn_data["network"]
    architecture = infer_architecture(network["weights"])

    lua_content = generate_lua(version, network, architecture, picker["description"])

    output_path = os.path.join(output_dir, picker["output"])
    with open(output_path, "w") as f:
        f.write(lua_content)

    arch_str = " -> ".join(str(s) for s in architecture)
    print(f"  {picker['output']}: {arch_str} ({version})")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert trained weight files to bundled Lua default weights"
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default=os.path.expanduser("~/.local/share/nvim/neural-open"),
        help="Directory containing weight files (default: ~/.local/share/nvim/neural-open)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Error: Directory not found: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    # Determine output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    output_dir = os.path.join(repo_root, "lua", "neural-open", "algorithms")

    print(f"Reading from: {args.input_dir}")
    print(f"Writing to:   {output_dir}")
    print()

    processed = 0
    for picker in PICKERS:
        if process_picker(args.input_dir, picker, output_dir):
            processed += 1

    if processed == 0:
        print("\nError: No weight files were processed", file=sys.stderr)
        sys.exit(1)

    print(f"\nProcessed {processed} picker(s)")


if __name__ == "__main__":
    main()
