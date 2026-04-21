"""Console entry points: ``cmb-diag tf-ee | tf-te | pol-angle | run``.

Each subcommand parses args, loads config, runs the relevant Pipeline step(s),
and writes .npz + .png artefacts under ``cfg.output_dir``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from cmb_diagnostics._types import Tracer
from cmb_diagnostics.config import Config
from cmb_diagnostics.estimators.base import FitResult
from cmb_diagnostics.pipeline import Pipeline
from cmb_diagnostics.reports import pol_angle as pa_reports
from cmb_diagnostics.reports import tf as tf_reports


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML config file (see docs/configuration.md).",
    )


def _ensure_output_dir(cfg: Config) -> Path:
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _save_tf(result: FitResult, out_dir: Path) -> None:
    tf_reports.save_npz(result, out_dir / f"{result.name}.npz")


def _save_pol_angle(result: FitResult, out_dir: Path) -> None:
    pa_reports.save_npz(result, out_dir / f"{result.name}.npz")


def cmd_tf_ee(args: argparse.Namespace) -> int:
    cfg = Config.from_yaml(args.config)
    out_dir = _ensure_output_dir(cfg)
    pipe = Pipeline(cfg)
    pipe.load_mask()
    pipe.build_fieldsets()
    pipe.compute_spectra()
    results: list[FitResult] = []
    for band in cfg.so.bands:
        target = Tracer(cfg.so.name, band.freq, spin=2)
        result = pipe.estimate_tf_ee(target=target)
        _save_tf(result, out_dir)
        results.append(result)
    tf_reports.plot(results, path=out_dir / "tf_ee.png")
    return 0


def cmd_tf_te(args: argparse.Namespace) -> int:
    cfg = Config.from_yaml(args.config)
    out_dir = _ensure_output_dir(cfg)
    pipe = Pipeline(cfg)
    pipe.load_mask()
    pipe.build_fieldsets()
    pipe.compute_spectra()
    results: list[FitResult] = []
    for band in cfg.so.bands:
        target = Tracer(cfg.so.name, band.freq, spin=2)
        result = pipe.estimate_tf_te(target=target)
        _save_tf(result, out_dir)
        results.append(result)
    tf_reports.plot(results, path=out_dir / "tf_te.png")
    return 0


def cmd_pol_angle(args: argparse.Namespace) -> int:
    cfg = Config.from_yaml(args.config)
    out_dir = _ensure_output_dir(cfg)
    pipe = Pipeline(cfg)
    pipe.load_mask()
    pipe.build_fieldsets()
    pipe.compute_spectra()
    result = pipe.estimate_pol_angle()
    _save_pol_angle(result, out_dir)
    pa_reports.plot(result, path=out_dir / "pol_angle.png")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    cfg = Config.from_yaml(args.config)
    _ensure_output_dir(cfg)
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
