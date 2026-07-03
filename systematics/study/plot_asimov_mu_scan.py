#!/usr/bin/env python3
"""Draw Asimov 2DeltaNLL curves for the signal-strength modifier.

The available combine outputs for this analysis contain expected
significances, not likelihood scan points.  For an Asimov data set with
mu_hat = 1, the local Gaussian approximation gives

    2DeltaNLL(mu) = ((mu - 1) / sigma_mu)^2,    sigma_mu = 1 / Z_A.

The script reads Z_A from combine/UParTv2_v3/significance*.csv and writes the
main curve plot plus a standalone legend PDF.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

try:
    import mplhep as hep
except ImportError as exc:
    raise SystemExit(
        "This plotting script needs mplhep. Run it in the analysis plotting "
        "environment that provides matplotlib, numpy, and mplhep."
    ) from exc


DEFAULT_INPUT_DIR = "/afs/ihep.ac.cn/users/y/yiyangzhao/Research/CMS_THU_Space/VVV/ScoutingVVVTools_new/combine/UParTv2_v3/"
DEFAULT_OUTPUT_DIR = "./asimov_mu_scan"

MU_HAT = 1.0
THRESHOLD = 1.0
THRESHOLD_COLOR = "#ff0000"


@dataclass(frozen=True)
class Curve:
    key: str
    label: str
    color: str
    significance: float

    @property
    def sigma_mu(self) -> float:
        return 1.0 / self.significance

    @property
    def fit_label(self) -> str:
        sigma = self.sigma_mu
        return (
            rf"$\hat{{\mu}}_{{\mathrm{{SM}}}} = "
            rf"{MU_HAT:.2f}^{{+{sigma:.2f}}}_{{-{sigma:.2f}}}$"
        )


def read_combined_significance(csv_path: Path) -> float:
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["scope"] == "combined" and row["name"] == "combined":
                return float(row["significance"])
    raise ValueError(f"Missing combined significance in {csv_path}")


def read_channel_significance(csv_path: Path, channel: str) -> float:
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if (
                row["channel"] == channel
                and row["scope"] == "combined"
                and row["name"] == "combined"
            ):
                return float(row["significance"])
    raise ValueError(f"Missing {channel} combined significance in {csv_path}")


def load_curves(input_dir: Path) -> list[Curve]:
    combined_csv = input_dir / "significance.csv"
    by_channel_csv = input_dir / "significance_by_channel.csv"

    return [
        Curve(
            key="fat2",
            label=r"$0\ell$-2J",
            color="#5790fc",
            significance=read_channel_significance(by_channel_csv, "fat2"),
        ),
        Curve(
            key="fat3",
            label=r"$0\ell$-3J",
            color="#f89c20",
            significance=read_channel_significance(by_channel_csv, "fat3"),
        ),
        Curve(
            key="combined",
            label="Full analysis",
            color="#e42536",
            significance=read_combined_significance(combined_csv),
        ),
    ]


def two_delta_nll(mu: np.ndarray, curve: Curve) -> np.ndarray:
    return ((mu - MU_HAT) / curve.sigma_mu) ** 2


def add_cms_label(ax: plt.Axes, lumi: float, com: float) -> None:
    try:
        hep.cms.label(label="", data=True, lumi=lumi, com=com, ax=ax)
    except TypeError:
        hep.cms.label("", data=True, lumi=lumi, com=com, ax=ax)


def draw_main_plot(
    curves: list[Curve],
    output_path: Path,
    lumi: float,
    com: float,
    mu_min: float,
    mu_max: float,
    y_max: float,
) -> None:
    hep.style.use("CMS")
    plt.rcParams.update(
        {
            "font.size": 22,
            "axes.labelsize": 30,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "legend.fontsize": 17,
            "figure.figsize": (8.8, 6.2),
            "figure.dpi": 120,
        }
    )

    mu = np.linspace(mu_min, mu_max, 1000)
    fig, ax = plt.subplots()

    for curve in curves:
        ax.plot(
            mu,
            two_delta_nll(mu, curve),
            color=curve.color,
            lw=3.0,
            solid_capstyle="round",
            label=curve.label,
        )

    ax.axhline(
        THRESHOLD,
        color=THRESHOLD_COLOR,
        lw=1.8,
        label=r"$2\Delta\mathrm{NLL}$ threshold",
    )

    ax.set_xlim(mu_min, mu_max)
    ax.set_ylim(0.0, y_max)
    ax.set_xlabel(r"SM ($\mu_{\mathrm{SM}}$)", ha="right", x=1.0)
    ax.set_ylabel(r"$2\Delta\mathrm{NLL}$")

    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.tick_params(which="major", length=8, width=1.4)
    ax.tick_params(which="minor", length=4, width=1.0)

    add_cms_label(ax, lumi=lumi, com=com)

    handles = [
        Line2D([0], [0], color=curve.color, lw=3.0, label=curve.label)
        for curve in curves
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            color=THRESHOLD_COLOR,
            lw=1.8,
            label=r"$2\Delta\mathrm{NLL}$ threshold",
        )
    )
    ax.legend(
        handles=handles,
        loc="lower left",
        frameon=False,
        handlelength=2.4,
        borderaxespad=0.4,
    )

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def draw_standalone_legend(curves: list[Curve], output_path: Path) -> None:
    hep.style.use("CMS")
    plt.rcParams.update({"font.size": 28, "figure.figsize": (10.4, 4.8)})

    fig, ax = plt.subplots()
    ax.set_axis_off()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    entries = [
        (curves[2].color, curves[2].label, curves[2].fit_label, 3.0),
        (
            THRESHOLD_COLOR,
            r"$2\Delta\mathrm{NLL}$ threshold",
            "68% CL",
            1.8,
        ),
        (curves[0].color, curves[0].label, curves[0].fit_label, 3.0),
        (curves[1].color, curves[1].label, curves[1].fit_label, 3.0),
    ]

    positions = [(0.06, 0.76), (0.56, 0.76), (0.06, 0.34), (0.56, 0.34)]
    for (color, title, subtitle, line_width), (x0, y0) in zip(entries, positions):
        ax.plot(
            [x0, x0 + 0.085],
            [y0, y0],
            color=color,
            lw=line_width,
            solid_capstyle="butt",
            transform=ax.transAxes,
            clip_on=False,
        )
        ax.text(
            x0 + 0.115,
            y0 + 0.075,
            title,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=30,
        )
        ax.text(
            x0 + 0.115,
            y0 - 0.075,
            subtitle,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=25,
        )

    """
    ax.add_patch(
        Rectangle(
            (0.01, 0.02),
            0.98,
            0.96,
            transform=ax.transAxes,
            fill=False,
            edgecolor="0.85",
            linewidth=1.0,
            clip_on=False,
        )
    )
    """

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw Asimov 2DeltaNLL curves for the VVV scouting analysis."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing significance.csv and significance_by_channel.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the PDF files are written.",
    )
    parser.add_argument(
        "--plot-name",
        default="asimov_mu_2dnll.pdf",
        help="Output filename for the main 2DeltaNLL plot.",
    )
    parser.add_argument(
        "--legend-name",
        default="asimov_mu_2dnll_legend.pdf",
        help="Output filename for the standalone legend.",
    )
    parser.add_argument("--lumi", type=float, default=256, help="Luminosity in fb^-1.")
    parser.add_argument("--com", type=float, default=13.6, help="Collision energy in TeV.")
    parser.add_argument("--mu-min", type=float, default=0.0, help="Minimum mu on the x axis.")
    parser.add_argument("--mu-max", type=float, default=2.1, help="Maximum mu on the x axis.")
    parser.add_argument("--y-max", type=float, default=2.5, help="Maximum 2DeltaNLL.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    curves = load_curves(args.input_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    draw_main_plot(
        curves=curves,
        output_path=args.output_dir / args.plot_name,
        lumi=args.lumi,
        com=args.com,
        mu_min=args.mu_min,
        mu_max=args.mu_max,
        y_max=args.y_max,
    )
    draw_standalone_legend(curves, args.output_dir / args.legend_name)

    print(f"Wrote {args.output_dir / args.plot_name}")
    print(f"Wrote {args.output_dir / args.legend_name}")
    for curve in curves:
        print(
            f"{curve.key}: Z={curve.significance:.5g}, "
            f"sigma_mu={curve.sigma_mu:.4f}"
        )


if __name__ == "__main__":
    main()
