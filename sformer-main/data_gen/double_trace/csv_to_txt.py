#!/usr/bin/env python3
"""Convert double_trace trace_data_1300.csv to pcb_traces.txt.

Input format (block style):
- Group N
- segments,"[...]"
- point1_list,"[(x, y), ...]"
- parallel_segments,"[...]"
- point2_list,"[(x, y), ...]"

Output format:
PCB_Index,Trace_Index,Node_Index,X,Y,Z,Layer
"""

from __future__ import annotations

import argparse
import ast
import csv
import re
from pathlib import Path

DEFAULT_INPUT = "/home/dengnuo/share/sformer/data/double_trace/trace_data_1300.csv"
DEFAULT_OUTPUT = "/home/dengnuo/share/sformer/data/double_trace/pcb_traces.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert trace_data_1300.csv to pcb_traces.txt")
    parser.add_argument("--input", type=Path, default=Path(DEFAULT_INPUT), help="Input trace_data_1300.csv path")
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT), help="Output pcb_traces.txt path")
    parser.add_argument("--z", type=float, default=1.0, help="Constant Z value for all points")
    parser.add_argument("--layer", type=int, default=1, help="Constant Layer value for all points")
    return parser.parse_args()


def parse_group_blocks(csv_text: str) -> list[tuple[int, list[tuple[float, float]], list[tuple[float, float]]]]:
    lines = csv_text.splitlines()
    records: list[tuple[int, list[tuple[float, float]], list[tuple[float, float]]]] = []

    group_id: int | None = None
    p1: list[tuple[float, float]] | None = None
    p2: list[tuple[float, float]] | None = None

    def flush_record() -> None:
        nonlocal group_id, p1, p2
        if group_id is not None and p1 is not None and p2 is not None:
            records.append((group_id, p1, p2))

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        group_match = re.match(r"^Group\s+(\d+)$", line)
        if group_match:
            flush_record()
            group_id = int(group_match.group(1))
            p1 = None
            p2 = None
            continue

        if line.startswith("point1_list,"):
            payload = line.split(",", 1)[1].strip()
            if payload.startswith('"') and payload.endswith('"'):
                payload = payload[1:-1]
            parsed = ast.literal_eval(payload)
            p1 = [(float(x), float(y)) for x, y in parsed]
            continue

        if line.startswith("point2_list,"):
            payload = line.split(",", 1)[1].strip()
            if payload.startswith('"') and payload.endswith('"'):
                payload = payload[1:-1]
            parsed = ast.literal_eval(payload)
            p2 = [(float(x), float(y)) for x, y in parsed]
            continue

    flush_record()
    return records


def write_pcb_traces_txt(
    output_path: Path,
    records: list[tuple[int, list[tuple[float, float]], list[tuple[float, float]]]],
    z_value: float,
    layer_value: int,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["PCB_Index", "Trace_Index", "Node_Index", "X", "Y", "Z", "Layer"])

        for pcb_index, trace1_points, trace2_points in records:
            for node_idx, (x, y) in enumerate(trace1_points):
                writer.writerow([pcb_index, 0, node_idx, x, y, z_value, layer_value])
                total_rows += 1

            for node_idx, (x, y) in enumerate(trace2_points):
                writer.writerow([pcb_index, 1, node_idx, x, y, z_value, layer_value])
                total_rows += 1

    return total_rows


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    csv_text = args.input.read_text(encoding="utf-8")
    records = parse_group_blocks(csv_text)

    if not records:
        raise RuntimeError("No valid group records parsed from input CSV")

    row_count = write_pcb_traces_txt(args.output, records, args.z, args.layer)

    print(f"Parsed groups: {len(records)}")
    print(f"Output rows: {row_count}")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
