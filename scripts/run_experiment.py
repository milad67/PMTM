"""Launch training with the recorded PMTM experiment arguments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys


def build_command(
    root: Path,
    dataset: str,
    datapath: str,
    output_dir: str,
    resume: str = "",
) -> list[str]:
    config = json.loads((root / "configs" / f"brats{dataset}.json").read_text())
    command = [
        sys.executable,
        str(root / "train.py"),
        "--datapath", datapath,
        "--output_dir", output_dir,
    ]
    if resume:
        command.extend(["--resume", resume])
    for key, value in config.items():
        if isinstance(value, bool):
            if value:
                command.append(f"--{key}")
        elif isinstance(value, (str, int, float)):
            command.extend([f"--{key}", str(value)])
        else:
            raise ValueError(f"Unsupported configuration value for {key!r}")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["2018", "2020", "2023"])
    parser.add_argument("--datapath", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument("--print-command", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    command = build_command(root, args.dataset, args.datapath, args.output_dir, args.resume)
    print(shlex.join(command), flush=True)
    if not args.print_command:
        subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
