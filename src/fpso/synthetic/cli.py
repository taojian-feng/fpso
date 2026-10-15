"""Generate mock extracts for FPSO Kestrel.

fpso-synth --seed 7 --profile A --out data/kestrel-A
"""

from __future__ import annotations

import argparse
import json

from fpso.synthetic.corrupt import PROFILES
from fpso.synthetic.export import export
from fpso.synthetic.truth import build_world


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--profile", choices=sorted(PROFILES), default="A")
    ap.add_argument("--scale", type=float, default=1.0, help="1.0 = full size from the plan")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    run = export(build_world(seed=args.seed, scale=args.scale), args.out, profile=args.profile)
    print(json.dumps(run["counts"], indent=2))


if __name__ == "__main__":
    main()
