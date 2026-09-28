from __future__ import annotations

import argparse

from .demo import run_demo


def main() -> None:
    parser = argparse.ArgumentParser(prog="robot-map", description="2D robot mapping platform")
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser("demo", help="run built-in true-loop and similar-corridor scenarios")
    demo.add_argument("project", help="project directory for scans, poses, events and map versions")
    args = parser.parse_args()
    if args.command == "demo":
        report = run_demo(args.project)
        print(f"demo report: {report}")


if __name__ == "__main__":
    main()
