"""Local shim for nemo-toolkit's manifest_utils.

Provides write_manifest and read_manifest so that MInference's per-task
scripts work without installing nemo-toolkit.
"""

import json


def write_manifest(output_path, target_manifest, ensure_ascii: bool = True):
    with open(output_path, "w", encoding="utf-8") as outfile:
        for tgt in target_manifest:
            json.dump(tgt, outfile, ensure_ascii=ensure_ascii)
            outfile.write("\n")


def read_manifest(input_path):
    entries = []
    with open(input_path, "r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries
