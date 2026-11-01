"""Quality gate: block the release if the exam scores drop below the line.

Usage:  python quality_gate.py            (reads results.json)
Exit code 0 = ship it, 1 = blocked. CI systems treat exit code 1 as a failed check.
"""
import json
import sys
from pathlib import Path

THRESHOLDS = {
    "search_hit_rate": 0.90,    # the right page must come back at least 90% of the time
    "answer_pass_rate": 0.90,   # answers must be faithful and relevant at least 90% of the time
}


def main():
    scores = json.loads(Path("results.json").read_text())["scores"]
    failed = False
    print("\nQuality gate")
    for name, minimum in THRESHOLDS.items():
        value = scores[name]
        ok = value >= minimum
        failed |= not ok
        print(f"  {'OK  ' if ok else 'FAIL'} {name}: {value:.2f} (minimum {minimum:.2f})")
    if failed:
        print("\nRelease BLOCKED: scores are below the line.\n")
        sys.exit(1)
    print("\nRelease APPROVED: all scores pass.\n")


if __name__ == "__main__":
    main()
