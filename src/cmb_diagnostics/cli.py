"""Console entry points: ``cmb-diag tf-ee | tf-te | pol-angle | run``.

Each subcommand is a thin wrapper: parse args -> Config.from_yaml -> Pipeline ->
call the corresponding step -> pass result to reports.*. In Phase 2 the inner
Pipeline call raises NotImplementedError; the wiring above is real.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cmb_diagnostics._types import Tracer
from cmb_diagnostics.config import Config
from cmb_diagnostics.pipeline import Pipeline


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML config file (see docs/configuration.md).",
    )


def cmd_tf_ee(args: argparse.Namespace) -> int:
    cfg = Config.from_yaml(args.config)
    pipe = Pipeline(cfg)
    pipe.load_mask()
    pipe.build_fieldsets()
    pipe.compute_spectra()
    for band in cfg.so.bands:
        target = Tracer(cfg.so.name, band.freq, spin=2)
        result = pipe.estimate_tf_ee(target=target)
        pipe.results[f"tf_ee_{cfg.so.name}_{int(band.freq)}"] = result
    return 0


def cmd_tf_te(args: argparse.Namespace) -> int:
    cfg = Config.from_yaml(args.config)
    pipe = Pipeline(cfg)
    pipe.load_mask()
    pipe.build_fieldsets()
    pipe.compute_spectra()
    for band in cfg.so.bands:
        target = Tracer(cfg.so.name, band.freq, spin=2)
        result = pipe.estimate_tf_te(target=target)
        pipe.results[f"tf_te_{cfg.so.name}_{int(band.freq)}"] = result
    return 0


def cmd_pol_angle(args: argparse.Namespace) -> int:
    cfg = Config.from_yaml(args.config)
    pipe = Pipeline(cfg)
    pipe.load_mask()
    pipe.build_fieldsets()
    pipe.compute_spectra()
    pipe.results["pol_angle"] = pipe.estimate_pol_angle()
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    cfg = Config.from_yaml(args.config)
    pipe = Pipeline(cfg)
    pipe.run()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cmb-diag",
        description="SO calibration diagnostics (transfer function, polarization angle).",
    )
    subs = p.add_subparsers(dest="command", required=True)

    p_tf_ee = subs.add_parser("tf-ee", help="Compute the EE transfer function.")
    _add_common(p_tf_ee)
    p_tf_ee.set_defaults(func=cmd_tf_ee)

    p_tf_te = subs.add_parser("tf-te", help="Compute the TE transfer function.")
    _add_common(p_tf_te)
    p_tf_te.set_defaults(func=cmd_tf_te)

    p_pa = subs.add_parser("pol-angle", help="Estimate the SO polarization angle.")
    _add_common(p_pa)
    p_pa.set_defaults(func=cmd_pol_angle)

    p_run = subs.add_parser("run", help="Run the full pipeline (tf-ee + tf-te + pol-angle).")
    _add_common(p_run)
    p_run.set_defaults(func=cmd_run)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
