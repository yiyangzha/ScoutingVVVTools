"""Summary: Compute AK4 tagger working points and efficiencies."""
import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

try:
    import mplhep as hep
except Exception:
    print("[FATAL] mplhep 未安装：pip install mplhep", file=sys.stderr)
    raise

try:
    import uproot
except Exception:
    print("[FATAL] uproot 未安装：pip install uproot", file=sys.stderr)
    raise

plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["mathtext.rm"] = "serif"
plt.style.use(hep.style.CMS)

ROOT_FILE = "b_veto_hists.root"
OUT_DIR = "b_veto"
os.makedirs(OUT_DIR, exist_ok=True)
CSV_FILE = os.path.join(OUT_DIR, "sample_btag_working_points.csv")
EVENT_COUNT_CSV_FILE = os.path.join(OUT_DIR, "top3_pass_count_efficiencies.csv")
SAMPLE_LABELS = ["WWW", "TTbar"]
#TARGET_LIGHT_EFFS = [0.8, 0.5, 0.4, 0.3, 0.2, 0.1, 0.01, 0.001]
TARGET_LIGHT_EFFS = [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1] + [0.01, 0.001]

_default_colors = plt.rcParams["axes.prop_cycle"].by_key().get(
    "color", ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
)
COLOR_MAP = {
    "S": _default_colors[2 % len(_default_colors)],
    "B": _default_colors[3 % len(_default_colors)],
}
LABEL_MAP = {
    "S": "Signal (WWW)",
    "B": "Background (TTbar)",
}
SAMPLE_COLOR_MAP = {
    "WWW": _default_colors[2 % len(_default_colors)],
    "TTbar": _default_colors[3 % len(_default_colors)],
}


def get_hist_arrays(hist):
    values, edges = hist.to_numpy(flow=False)
    return values.astype(float), edges.astype(float)


def get_hist2d_arrays(hist):
    values, x_edges, y_edges = hist.to_numpy(flow=False)
    return values.astype(float), x_edges.astype(float), y_edges.astype(float)


def get_hist3d_arrays(hist):
    values, x_edges, y_edges, z_edges = hist.to_numpy(flow=False)
    return (
        values.astype(float),
        x_edges.astype(float),
        y_edges.astype(float),
        z_edges.astype(float),
    )


def normalize_counts(counts):
    total = counts.sum()
    if total <= 0:
        return counts * 0.0
    return counts / total


def build_cumulative_eff(counts):
    total = counts.sum()
    if total <= 0:
        return np.zeros_like(counts, dtype=float)
    return np.cumsum(counts) / total


def rebin_hist(counts, edges, target_bins=50):
    if counts is None or edges is None:
        return None, None

    nbins = len(counts)
    if nbins == 0:
        return counts.copy(), edges.copy()

    if nbins % target_bins == 0:
        factor = nbins // target_bins
        counts_r = counts.reshape(target_bins, factor).sum(axis=1)
        edges_r = np.linspace(edges[0], edges[-1], target_bins + 1)
        return counts_r, edges_r

    edges_r = np.linspace(edges[0], edges[-1], target_bins + 1)
    cdf_old = np.r_[0.0, np.cumsum(counts)]
    cdf_new = np.interp(edges_r, edges, cdf_old)
    counts_r = np.diff(cdf_new)
    return counts_r, edges_r


def get_object(root_obj, path):
    obj = root_obj
    for part in path.split("/"):
        obj = obj[part]
    return obj


def load_hist(root_file, candidates):
    for name in candidates:
        try:
            hist = get_object(root_file, name)
            return get_hist_arrays(hist), name
        except KeyError:
            continue
    raise RuntimeError(f"找不到任何候选直方图: {candidates}")


def load_hist2d(root_file, candidates):
    for name in candidates:
        try:
            hist = get_object(root_file, name)
            return get_hist2d_arrays(hist), name
        except KeyError:
            continue
    raise RuntimeError(f"找不到任何候选二维直方图: {candidates}")


def load_hist3d(root_file, candidates):
    for name in candidates:
        try:
            hist = get_object(root_file, name)
            return get_hist3d_arrays(hist), name
        except KeyError:
            continue
    raise RuntimeError(f"找不到任何候选三维直方图: {candidates}")


def read_inputs(root_path):
    print(f"[INFO] 打开 ROOT 文件: {root_path}")
    root_file = uproot.open(root_path)

    inputs = {}
    for cat_tag in ["cat2", "cat3"]:
        (sig_counts, sig_edges), sig_name = load_hist(
            root_file,
            [f"h_signal_{cat_tag}", f"per_sample/h_minBtag_{cat_tag}_WWW"],
        )
        (bkg_counts, bkg_edges), bkg_name = load_hist(
            root_file,
            [f"h_background_{cat_tag}", f"per_sample/h_minBtag_{cat_tag}_TTbar"],
        )

        if not np.allclose(sig_edges, bkg_edges, rtol=0, atol=1e-12):
            raise RuntimeError(f"{cat_tag} 的 signal/background binning 不一致。")

        inputs[cat_tag] = {
            "S": (sig_counts, sig_edges),
            "B": (bkg_counts, bkg_edges),
        }

        print(
            f"[INFO] {cat_tag}: "
            f"signal <- {sig_name}, background <- {bkg_name}"
        )

    return inputs


def read_sample_flavour_inputs(root_path):
    print(f"[INFO] 读取 sample 级二维直方图: {root_path}")
    root_file = uproot.open(root_path)

    sample_inputs = {}
    for sample in SAMPLE_LABELS:
        (values, x_edges, y_edges), hist_name = load_hist2d(
            root_file,
            [f"h2_probb_vs_hadronFlavour_{sample}"],
        )
        sample_inputs[sample] = {
            "values": values,
            "x_edges": x_edges,
            "y_edges": y_edges,
        }
        print(f"[INFO] {sample}: flavour map <- {hist_name}")

    return sample_inputs


def read_top3_inputs(root_path):
    print(f"[INFO] 读取 sample 级 top3 三维直方图: {root_path}")
    root_file = uproot.open(root_path)

    top3_inputs = {}
    for sample in SAMPLE_LABELS:
        top3_inputs[sample] = {}
        for cat_tag in ["cat2", "cat3"]:
            (values, x_edges, y_edges, z_edges), hist_name = load_hist3d(
                root_file,
                [
                    f"h3_top3_probb_{cat_tag}_{sample}",
                    f"per_sample/h3_top3_probb_{cat_tag}_{sample}",
                ],
            )
            top3_inputs[sample][cat_tag] = {
                "values": values,
                "x_edges": x_edges,
                "y_edges": y_edges,
                "z_edges": z_edges,
            }
            print(f"[INFO] {sample} {cat_tag}: top3 map <- {hist_name}")

    return top3_inputs


def extract_flavour_counts(hist_values, y_edges, flavour):
    idx = np.searchsorted(y_edges, flavour, side="right") - 1
    if idx < 0 or idx >= len(y_edges) - 1:
        raise RuntimeError(f"hadronFlavour={flavour} 不在二维图 y 轴范围内。")
    if not (y_edges[idx] <= flavour < y_edges[idx + 1]):
        raise RuntimeError(f"无法在二维图中定位 hadronFlavour={flavour} 的 bin。")
    return hist_values[:, idx].astype(float)


def threshold_for_tail_eff(counts, edges, target_eff):
    total = counts.sum()
    if total <= 0:
        return np.nan
    if target_eff >= 1.0:
        return float(edges[0])
    if target_eff <= 0.0:
        return float(edges[-1])

    survival_low = np.cumsum(counts[::-1])[::-1] / total

    for idx, s_low in enumerate(survival_low):
        s_high = survival_low[idx + 1] if idx + 1 < len(survival_low) else 0.0
        if s_low >= target_eff >= s_high:
            bin_content = s_low - s_high
            if bin_content <= 0.0:
                return float(edges[idx])

            frac_above = (target_eff - s_high) / bin_content
            width = edges[idx + 1] - edges[idx]
            threshold = edges[idx + 1] - frac_above * width
            return float(np.clip(threshold, edges[idx], edges[idx + 1]))

    return float(edges[-1])


def tail_efficiency(counts, edges, threshold):
    total = counts.sum()
    if total <= 0 or not np.isfinite(threshold):
        return np.nan

    if threshold <= edges[0]:
        return 1.0
    if threshold >= edges[-1]:
        return 0.0

    idx = np.searchsorted(edges, threshold, side="right") - 1
    if idx < 0:
        return 1.0
    if idx >= len(counts):
        return 0.0

    upper_tail = counts[idx + 1 :].sum()
    width = edges[idx + 1] - edges[idx]
    frac_above = 0.0 if width <= 0 else (edges[idx + 1] - threshold) / width
    return float((upper_tail + frac_above * counts[idx]) / total)


def format_csv_value(value):
    if isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if value is None or not np.isfinite(value):
        return ""
    return f"{value:.8f}"


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[SAVE] {path}")


def compute_sample_working_points(sample_inputs):
    rows = []
    threshold_map = {sample: {} for sample in SAMPLE_LABELS}

    for sample in SAMPLE_LABELS:
        info = sample_inputs[sample]
        light_counts = extract_flavour_counts(info["values"], info["y_edges"], 0)
        b_counts = extract_flavour_counts(info["values"], info["y_edges"], 5)

        light_total = float(light_counts.sum())
        b_total = float(b_counts.sum())

        for target_eff in TARGET_LIGHT_EFFS:
            threshold = threshold_for_tail_eff(light_counts, info["x_edges"], target_eff)
            actual_light_eff = tail_efficiency(light_counts, info["x_edges"], threshold)
            b_eff = tail_efficiency(b_counts, info["x_edges"], threshold)
            threshold_map[sample][target_eff] = threshold

            rows.append(
                {
                    "sample": sample,
                    "light_jet_efficiency_target": format_csv_value(target_eff),
                    "probb_threshold": format_csv_value(threshold),
                    "actual_light_jet_efficiency": format_csv_value(actual_light_eff),
                    "b_jet_efficiency": format_csv_value(b_eff),
                    "light_jet_entries": format_csv_value(light_total),
                    "b_jet_entries": format_csv_value(b_total),
                }
            )

    return rows, threshold_map


def write_sample_working_points(rows):
    write_csv(
        CSV_FILE,
        [
            "sample",
            "light_jet_efficiency_target",
            "probb_threshold",
            "actual_light_jet_efficiency",
            "b_jet_efficiency",
            "light_jet_entries",
            "b_jet_entries",
        ],
        rows,
    )


def axis_centers(edges):
    return 0.5 * (edges[:-1] + edges[1:])


def compute_top3_pass_count_distribution(top3_info, threshold):
    values = top3_info["values"]
    x_pass = axis_centers(top3_info["x_edges"]) > threshold
    y_pass = axis_centers(top3_info["y_edges"]) > threshold
    z_pass = axis_centers(top3_info["z_edges"]) > threshold

    pass_counts = (
        x_pass[:, None, None].astype(np.int8)
        + y_pass[None, :, None].astype(np.int8)
        + z_pass[None, None, :].astype(np.int8)
    )

    return np.array(
        [values[pass_counts == n_pass].sum() for n_pass in range(4)],
        dtype=float,
    )


def threshold_tag(target_eff):
    return f"{target_eff:g}".replace(".", "p")


def plot_top3_pass_count_distributions(top3_inputs, ttbar_thresholds):
    for target_eff in TARGET_LIGHT_EFFS:
        threshold = ttbar_thresholds[target_eff]
        if not np.isfinite(threshold):
            print(f"[WARN] TTbar 在 light-jet efficiency={target_eff:g} 时阈值无效，跳过该图。")
            continue

        fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharey=True)
        xvals = np.arange(4)
        plotted_any = False

        for ax, cat_tag, title in zip(
            axes,
            ["cat2", "cat3"],
            [r"2 AK8", r"$\geq$3 AK8"],
        ):
            for sample in SAMPLE_LABELS:
                dist = compute_top3_pass_count_distribution(top3_inputs[sample][cat_tag], threshold)
                total = dist.sum()
                if total <= 0:
                    print(f"[WARN] {sample} {cat_tag} 在阈值 {threshold:.6f} 下没有 event，跳过该曲线。")
                    continue

                yvals = dist / total
                ax.plot(
                    xvals,
                    yvals,
                    marker="o",
                    lw=1.8,
                    color=SAMPLE_COLOR_MAP[sample],
                    label=sample,
                )
                plotted_any = True

            ax.set_title(title)
            ax.set_xlabel(r"$N(\mathrm{AK4})$")
            ax.set_xticks(xvals)
            ax.set_xlim(-0.1, 3.1)
            ax.set_ylim(0.0, 1.0)
            ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.6)

        if not plotted_any:
            plt.close(fig)
            continue

        axes[0].set_ylabel("Event Fraction")
        axes[1].legend(frameon=False)
        #hep.cms.label(data=False, com=13.6, year="2024", ax=axes[0])
        fig.suptitle(
            r"$\epsilon^{light}=$" + f"{target_eff:g}",
            y=0.95,
        )

        out_path = os.path.join(
            OUT_DIR,
            f"N_AK4_{threshold_tag(target_eff)}.pdf",
        )
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        print(f"[SAVE] {out_path}")


def build_top3_efficiency_rows(top3_inputs, ttbar_thresholds):
    rows = []

    for target_eff in TARGET_LIGHT_EFFS:
        threshold = ttbar_thresholds[target_eff]
        for sample in SAMPLE_LABELS:
            for cat_tag in ["cat2", "cat3"]:
                dist = compute_top3_pass_count_distribution(top3_inputs[sample][cat_tag], threshold)
                total = dist.sum()
                eff_eq0 = dist[0] / total if total > 0 else np.nan
                eff_eq1 = dist[1] / total if total > 0 else np.nan

                rows.append(
                    {
                        "threshold_source_sample": "TTbar",
                        "light_jet_efficiency_target": format_csv_value(target_eff),
                        "probb_threshold": format_csv_value(threshold),
                        "sample": sample,
                        "category": cat_tag,
                        "total_events": format_csv_value(total),
                        "n_pass_eq_0_events": format_csv_value(dist[0]),
                        "n_pass_eq_1_events": format_csv_value(dist[1]),
                        "n_pass_eq_2_events": format_csv_value(dist[2]),
                        "n_pass_eq_3_events": format_csv_value(dist[3]),
                        "n_pass_eq_0_efficiency": format_csv_value(eff_eq0),
                        "n_pass_eq_1_efficiency": format_csv_value(eff_eq1),
                    }
                )

    return rows


def write_top3_efficiency_csv(rows):
    write_csv(
        EVENT_COUNT_CSV_FILE,
        [
            "threshold_source_sample",
            "light_jet_efficiency_target",
            "probb_threshold",
            "sample",
            "category",
            "total_events",
            "n_pass_eq_0_events",
            "n_pass_eq_1_events",
            "n_pass_eq_2_events",
            "n_pass_eq_3_events",
            "n_pass_eq_0_efficiency",
            "n_pass_eq_1_efficiency",
        ],
        rows,
    )


def plot_overlays(inputs):
    for cat_tag in ["cat2", "cat3"]:
        fig, ax = plt.subplots(figsize=(10, 8))
        plotted_any = False

        for key in ["S", "B"]:
            counts, edges = inputs[cat_tag][key]
            if counts.sum() <= 0:
                print(f"[WARN] {cat_tag} 的 {LABEL_MAP[key]} 为空，跳过该曲线。")
                continue

            counts_r, edges_r = rebin_hist(counts, edges, target_bins=50)
            yvals = normalize_counts(counts_r)
            ax.step(
                edges_r[:-1],
                yvals,
                where="post",
                lw=1.8,
                color=COLOR_MAP[key],
                label=LABEL_MAP[key],
            )
            plotted_any = True

        if not plotted_any:
            plt.close(fig)
            print(f"[WARN] {cat_tag} 没有可绘制的分布，跳过叠加图。")
            continue

        ax.set_xlabel("max(AK4 UParT b-score)")
        ax.set_ylabel("A.U.")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(1e-5, 1.0)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.6)
        ax.legend(frameon=False)

        hep.cms.label("Preliminary", data=False, com=13.6, year="2024", ax=ax)
        out_path = os.path.join(OUT_DIR, f"maxBtag_{cat_tag}.pdf")
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        print(f"[SAVE] {out_path}")


def plot_roc(inputs):
    fig, ax = plt.subplots(figsize=(10, 10))

    for cat_tag, style, label in [
        ("cat2", "-", r"2 AK8"),
        ("cat3", "--", r"$\geq$3 AK8"),
    ]:
        sig_counts, _ = inputs[cat_tag]["S"]
        bkg_counts, _ = inputs[cat_tag]["B"]

        if sig_counts.sum() <= 0 or bkg_counts.sum() <= 0:
            print(f"[WARN] {cat_tag} 的 signal/background 分布为空，跳过 ROC 曲线。")
            continue

        sig_eff = np.r_[build_cumulative_eff(sig_counts), 1.0]
        bkg_eff = np.r_[build_cumulative_eff(bkg_counts), 1.0]
        ax.plot(sig_eff, bkg_eff, style, lw=2.0, label=label)

    ax.plot([0, 1], [0, 1], color="k", lw=1.0, alpha=0.4, linestyle="--")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"$\epsilon_{Signal}$")
    ax.set_ylabel(r"$\epsilon_{Background}$")
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.6)
    ax.legend(frameon=False, loc="lower right")

    hep.cms.label("Preliminary", data=False, com=13.6, year="2024", ax=ax)
    out_path = os.path.join(OUT_DIR, "roc.pdf")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[SAVE] {out_path}")


def main():
    if not os.path.exists(ROOT_FILE):
        print(f"[FATAL] 找不到 {ROOT_FILE}，请先运行 b_veto.C 生成。", file=sys.stderr)
        sys.exit(1)

    inputs = read_inputs(ROOT_FILE)
    sample_inputs = read_sample_flavour_inputs(ROOT_FILE)
    top3_inputs = read_top3_inputs(ROOT_FILE)
    print("[INFO] 计算 sample 级 working points 并导出 CSV …")
    working_point_rows, threshold_map = compute_sample_working_points(sample_inputs)
    write_sample_working_points(working_point_rows)

    if "TTbar" not in threshold_map:
        raise RuntimeError("找不到 TTbar sample 的阈值，无法继续做 event-level top3 统计。")

    print("[INFO] 使用 TTbar 阈值计算 top3 AK4 通过数量分布与 event efficiency …")
    plot_top3_pass_count_distributions(top3_inputs, threshold_map["TTbar"])
    top3_eff_rows = build_top3_efficiency_rows(top3_inputs, threshold_map["TTbar"])
    write_top3_efficiency_csv(top3_eff_rows)
    print("[INFO] 绘制归一化叠加分布…")
    plot_overlays(inputs)
    print("[INFO] 绘制 ROC …")
    plot_roc(inputs)
    print("[DONE] 全部完成。输出目录：", OUT_DIR)


if __name__ == "__main__":
    main()
