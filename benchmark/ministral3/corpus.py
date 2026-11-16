"""Recreate the four 80-request manifests used in the original H200 comparison."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def build(output: Path) -> dict[str, str]:
    output.mkdir(parents=True, exist_ok=False)
    fixtures = Path(__file__).resolve().parents[2] / "test/manual/bagel"
    images = ("bagel.png", "bagel_gen.png")
    for name in images:
        shutil.copyfile(fixtures / name, output / name)
    questions = (
        "Explain how a rainbow forms.",
        "Explain why leaves change color in autumn.",
        "Describe how a bicycle stays balanced.",
        "Explain how a refrigerator cools food.",
        "Describe how plants turn sunlight into energy.",
        "Explain why the Moon has phases.",
        "Describe the steps in making bread.",
        "Explain how a compass works.",
    )
    workloads = {name: [] for name in ("short", "long", "image", "mixed")}
    for i in range(80):
        short = {"prompt": f"Request {i:03d}. " + questions[i % len(questions)]
                 + " Give a clear explanation with concrete details."}
        records = "\n".join(
            f"Day {j}: station {(i + j) % 11} recorded {18 + (i + j) % 13} degrees "
            f"and {(i * j) % 17} millimeters of rain." for j in range(90)
        )
        long = {"prompt": f"Report {i:03d}. Read the observations below.\n{records}\n"
                "Summarize the patterns and explain the limitations of these observations."}
        image = {
            "prompt": f"Image review {i:03d}. Describe the main subject, colors and arrangement in this image. "
                      "Use concrete details.",
            "image_path": images[i % 2],
        }
        for name, row in zip(workloads, (short, long, image, (short, long, image)[i % 3]), strict=True):
            workloads[name].append(row)
    for name, rows in workloads.items():
        (output / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(output.iterdir())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New directory for manifests and image fixtures")
    print(json.dumps(build(parser.parse_args().output), indent=2))
