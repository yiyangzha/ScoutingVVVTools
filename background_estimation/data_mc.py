#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare data and MC distributions
"""

import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import uproot

# -------------------- 可调参数（保持与 C++ 输出一致） --------------------
# 分支列表（用于判定 branch 名：hname 需以 "_{branch}" 结尾）
BRANCHES_FAT2 = [
    "N_ak8", "N_ak4", "H_T",
    "pt8_1", "pt8_2", "eta8_1", "eta8_2", "msd8_1", "msd8_2", "mr8_1", "mr8_2",
    "WvsQCD_1", "WvsQCD_2",
    "pt4_1", "pt4_2", "pt4_3", "pt4_4", "eta4_1", "eta4_2", "eta4_3", "eta4_4",
    "mPF4_1", "mPF4_2", "mPF4_3", "mPF4_4", "nConst4_1", "nConst4_2",
    "nConst4_3", "nConst4_4", "nCh4_1", "nCh4_2", "nCh4_3", "nCh4_4",
    "nEle4_1", "nEle4_2", "nEle4_3", "nEle4_4",
    "area4_1", "area4_2", "area4_3", "area4_4",
    "chEmEF4_1", "chEmEF4_2", "chEmEF4_3", "chEmEF4_4",
    "chHEF4_1", "chHEF4_2", "chHEF4_3", "chHEF4_4",
    "neEmEF4_1", "neEmEF4_2", "neEmEF4_3", "neEmEF4_4",
    "neHEF4_1", "neHEF4_2", "neHEF4_3", "neHEF4_4",
    "PT", "dR8", "dPhi", "m1overM", "m2overM", "sphereM",
    "dR84_min", "dR44_min",
    "ptL_1", "ptL_2", "ptL_3",
    "isoEcalL_1", "isoEcalL_2", "isoEcalL_3", "isoHcalL_1", "isoHcalL_2", "isoHcalL_3"
]

BRANCHES_FAT3 = [
    "N_ak8", "N_ak4", "H_T",
    "pt8_1", "pt8_2", "pt8_3", "eta8_1", "eta8_2", "eta8_3",
    "msd8_1", "msd8_2", "msd8_3", "mr8_1", "mr8_2", "mr8_3",
    "WvsQCD_1", "WvsQCD_2", "WvsQCD_3",
    "sphereM", "M", "m1overM", "m2overM", "m3overM", "PT",
    "dR_min", "dR_max", "dPhi_min", "dPhi_max",
    "ptL_1", "ptL_2", "ptL_3",
    "isoEcalL_1", "isoEcalL_2", "isoEcalL_3", "isoHcalL_1", "isoHcalL_2", "isoHcalL_3"
]

# MC 进程合并顺序与颜色（颜色可自行调整）
PROCESS_GROUPS = ['VVV', 'VH', 'VV', 'QCD', 'TT']
# 使用当前样式色板
_default_colors = plt.rcParams['axes.prop_cycle'].by_key().get(
    'color', ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd']
)
COLOR_MAP = {proc: _default_colors[i % len(_default_colors)] for i, proc in enumerate(PROCESS_GROUPS)}

# 图像保存目录
OUT_DIR = "pre-selection"

# -------------------- 样式设置 --------------------
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'
plt.style.use(hep.style.CMS)  # CMS 样式

# -------------------- 工具函数 --------------------
def key_to_group(key: str) -> str | None:
    """把 key（不区分大小写）映射到 VVV/VH/VV/QCD/TT 或 'data'；未知返回 None（忽略）。"""
    k = key.lower()
    if k == "data":
        return "data"
    if k in {"www", "wwz", "wzz", "zzz"}:
        return "VVV"
    if k in {"wplush", "wminush", "zh"}:
        return "VH"
    if k in {"ww", "wz", "zz"}:
        return "VV"
    if k.startswith("qcd"):
        return "QCD"
    if k.startswith("tt"):
        return "TT"
    return None

def th1_to_numpy(h):
    """从 uproot 的 TH1 对象取 (values, variances, edges)。若 variances 不存在则用 values 近似泊松误差。"""
    vals, edges = h.to_numpy()  # values already scaled
    vars_ = h.variances()
    if vars_ is None:
        vars_ = np.maximum(vals, 0.0)  # Poisson 近似（若已缩放，这里等价于 Sumw2 情形）
    return vals.astype(float), np.asarray(vars_, dtype=float), edges.astype(float)

def safe_add(to_vals, to_vars, vals, vars_):
    """将 (vals, vars_) 加到 (to_vals, to_vars) 上（就地）。"""
    to_vals += vals
    to_vars += vars_

def endswith_branch(hname: str, branches: list[str]) -> str | None:
    """在给定分支列表中找到使 hname 以 '_branch' 结尾的分支；返回 branch 或 None。"""
    for b in branches:
        if hname.endswith("_" + b):
            return b
    return None

def calc_ratio_mc_over_data(mc_vals, mc_vars, data_vals, data_vars):
    """R = MC/Data 的点估计与误差（误差传递）。"""
    mc_vals  = np.asarray(mc_vals,  dtype=float)
    mc_vars  = np.asarray(mc_vars,  dtype=float)
    data_vals= np.asarray(data_vals,dtype=float)
    data_vars= np.asarray(data_vars,dtype=float)

    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.where(data_vals > 0, mc_vals / data_vals, np.nan)
        term_mc = np.where(mc_vals   > 0, mc_vars  / np.maximum(mc_vals, 1e-300)**2, 0.0)
        term_dt = np.where(data_vals > 0, data_vars/ np.maximum(data_vals,1e-300)**2, 0.0)
        sigma_r = np.abs(r) * np.sqrt(term_mc + term_dt)
    return r, sigma_r

def first_last_true(mask: np.ndarray):
    """返回 mask 中 True 的首末索引（闭区间），若全 False 返回 (None, None)。"""
    idx = np.nonzero(mask)[0]
    if idx.size == 0:
        return None, None
    return int(idx[0]), int(idx[-1])

def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def pretty_ylim_max(ymax: float) -> float:
    """给 log y 取一个略有余量的上限。"""
    if not np.isfinite(ymax) or ymax <= 0:
        return 1.0
    return 10 ** (math.log10(ymax) + 0.15)

# -------------------- 主绘图函数 --------------------
def plot_all(rootfile: str):
    ensure_dir(OUT_DIR)

    with uproot.open(rootfile) as f:
        # lumi 从 counts 树读取（取首个/唯一值）
        lumi = None
        if "counts" in f:
            tree = f["counts"]
            try:
                arr = tree["lumi"].array(library="np")
                if arr.size > 0:
                    lumi = float(arr[0])
            except Exception:
                pass
        if lumi is None:
            lumi = 0.0

        for tdir_name, branches in [("fat2", BRANCHES_FAT2), ("fat3", BRANCHES_FAT3)]:
            if tdir_name not in f:
                print(f"[WARN] '{tdir_name}' not found in file, skip.")
                continue

            tdir = f[tdir_name]

            # 预聚合结构： per-branch -> per-group -> (vals, vars, edges)
            per_branch_data = {}
            for b in branches:
                per_branch_data[b] = {"data": None}
                for g in PROCESS_GROUPS:
                    per_branch_data[b][g] = None

            # 遍历该目录下的所有 TH1
            for key in tdir.keys(filter_classname="TH1*"):
                hname = key.split(";")[0]  # 去掉循环号
                branch = endswith_branch(hname, branches)
                if branch is None:
                    continue
                # 拆 key 与 branch（hname = "{key}_{branch}"，其中 branch 在末尾）
                key_part = hname[: -(len(branch) + 1)]
                grp = key_to_group(key_part)
                if grp is None:
                    continue

                h = tdir[hname]
                vals, vars_, edges = th1_to_numpy(h)

                # 统一 edges（同一 branch 内应一致）
                slot = per_branch_data[branch].get(grp, None)
                if slot is None:
                    per_branch_data[branch][grp] = (vals.copy(), vars_.copy(), edges)
                else:
                    v0, w0, e0 = slot
                    if not np.allclose(edges, e0):
                        print(f"[WARN] edges mismatch for {tdir_name}:{hname}, using first edges.")
                    safe_add(v0, w0, vals, vars_)
                    per_branch_data[branch][grp] = (v0, w0, e0)

            # 逐 branch 绘图
            for branch, grp_dict in per_branch_data.items():
                # 取出各组与 Data
                data_slot = grp_dict.get("data", None)
                # 统一使用某个已存在组的 edges
                edges = None
                for g in ["data"] + PROCESS_GROUPS:
                    slot = grp_dict.get(g, None)
                    if slot is not None:
                        edges = slot[2]
                        break
                if edges is None:
                    continue

                # 组装 MC 分组（vals/vars），并计算总 MC
                mc_vals_total = np.zeros(len(edges) - 1, dtype=float)
                mc_vars_total = np.zeros(len(edges) - 1, dtype=float)
                mc_group_vals = {}
                mc_group_vars = {}
                mc_yields = {}

                for g in PROCESS_GROUPS:
                    slot = grp_dict.get(g, None)
                    if slot is None:
                        vals_g = np.zeros_like(mc_vals_total)
                        vars_g = np.zeros_like(mc_vars_total)
                    else:
                        vals_g, vars_g, _ = slot
                    mc_group_vals[g] = vals_g
                    mc_group_vars[g] = vars_g
                    mc_vals_total += vals_g
                    mc_vars_total += vars_g
                    mc_yields[g] = float(np.sum(vals_g))

                mc_sigma = np.sqrt(np.maximum(mc_vars_total, 0.0))

                # Data
                if data_slot is None:
                    data_vals = np.zeros_like(mc_vals_total)
                    data_vars = np.zeros_like(mc_vars_total)
                else:
                    data_vals, data_vars, _ = data_slot
                data_sigma = np.sqrt(np.maximum(data_vars, 0.0))

                bin_centers = 0.5 * (edges[:-1] + edges[1:])
                bin_widths  = (edges[1:] - edges[:-1])

                # 根据 ≥0.1 的区间设置 xlim（至少 Data 或 MC 有一方 ≥ 0.1）
                mask_vis = (mc_vals_total >= 0.1) | (data_vals >= 0.1)
                i0, i1 = first_last_true(mask_vis)
                if i0 is None:
                    continue
                xlo, xhi = float(edges[i0]), float(edges[i1 + 1])

                # 堆叠顺序：少在下、多在上（按总产额升序）
                order = np.argsort([mc_yields[g] for g in PROCESS_GROUPS])
                groups_ordered = [PROCESS_GROUPS[i] for i in order]

                # ---- 作图 ----
                fig, (ax, axr) = plt.subplots(
                    2, 1, figsize=(10, 10),  # ×1.5 的画布
                    gridspec_kw={'height_ratios': [3, 1], 'hspace': 0},
                    sharex=True
                )

                # (A) MC 堆叠（去掉白缝 + 半透明）
                bottom = np.zeros_like(mc_vals_total)
                for g in groups_ordered:
                    vals_g = mc_group_vals[g]
                    color = COLOR_MAP[g]
                    ax.bar(
                        edges[:-1], vals_g, width=bin_widths, bottom=bottom,
                        align='edge', color=color, edgecolor='none',
                        linewidth=0, antialiased=False, alpha=0.9, label=g
                    )
                    bottom += vals_g
                ax.margins(x=0)  # 去掉两端额外留白，避免“缝隙”观感

                # (B) MC 统计不确定性带（灰色斜线）
                lower = np.clip(mc_vals_total - mc_sigma, 1e-12, None)
                upper = np.clip(mc_vals_total + mc_sigma, 1e-12, None)
                ax.fill_between(
                    bin_centers, lower, upper, step='mid',
                    facecolor='none', edgecolor='gray', hatch='///', linewidth=0
                )

                # (C) Data 黑点（仅纵向误差棒，marker 翻倍，误差棒加粗 1.5x）
                y_plot = np.where(data_vals > 0, data_vals, np.nan)
                ax.errorbar(
                    bin_centers, y_plot,
                    yerr=data_sigma, xerr=None,
                    fmt='o', ms=7.6, color='black', mfc='black', mec='black',
                    elinewidth=1.5, capsize=0, label='Data'
                )

                # 轴与标签
                ax.set_yscale('log')
                ax.set_xlim(xlo, xhi)
                ymax_combined = max(
                    np.nanmax(mc_vals_total[mask_vis]) if np.any(mask_vis) else 1.0,
                    np.nanmax(data_vals[mask_vis]) if np.any(mask_vis) else 1.0,
                    1.0
                )
                # 按要求：上 panel y 上限 = 两者最大值 * 5
                ax.set_ylim(0.1, max(1.0, ymax_combined * 5.0))
                ax.set_ylabel("Events", fontsize=24)
                # CMS 标注
                hep.cms.label("Preliminary", data=True, com=13.6, year="2024", lumi=lumi, ax=ax)

                # 图例（Data 放最后能覆盖），字号 +6
                handles, labels = ax.get_legend_handles_labels()
                if 'Data' in labels:
                    idx_data = labels.index('Data')
                    handles.append(handles.pop(idx_data))
                    labels.append(labels.pop(idx_data))
                ax.legend(handles, labels, loc='best', fontsize=17, frameon=False, ncol=2)

                # (D) 比值 MC/Data
                ratio, ratio_err = calc_ratio_mc_over_data(mc_vals_total, mc_vars_total, data_vals, data_vars)
                sel = (bin_centers >= xlo) & (bin_centers <= xhi)
                r_centers = bin_centers[sel]
                r_vals    = ratio[sel]
                r_errs    = ratio_err[sel]

                # 下方 ratio 点：marker 翻倍、误差棒加粗 1.5x
                axr.errorbar(
                    r_centers, r_vals, yerr=r_errs, xerr=None,
                    fmt='o', ms=7.6, color='black', mfc='black', mec='black',
                    elinewidth=1.5, capsize=0
                )
                # 基准线加粗 1.5x
                axr.axhline(1.0, color='black', linestyle='--', linewidth=1.5)

                # y 范围：若“数据最大值（含误差）”没到 5，则用 *1.5；否则上限=5
                rmax = np.nanmax(r_vals + np.nan_to_num(r_errs, nan=0.0))
                rmin = np.nanmin(r_vals - np.nan_to_num(r_errs, nan=0.0))
                if not np.isfinite(rmax) or rmax <= 0:
                    rmax = 1.0
                if rmax < 5.0:
                    axr.set_ylim(0.8 * rmin, 1.2 * rmax)
                else:
                    axr.set_ylim(0.0, 5.0)

                axr.set_ylabel(r'$\frac{MC}{Data}$', fontsize=26)
                #将纵坐标label向下移动
                axr.yaxis.set_label_coords(-0.05, 0.6)
                axr.set_xlabel(branch, fontsize=24)

                # 保存
                ensure_dir(OUT_DIR)
                out_name = os.path.join(OUT_DIR, f"{tdir_name}_{branch}.pdf")
                fig.savefig(out_name, dpi=300, bbox_inches='tight')
                plt.close(fig)

                print(f"[SAVE] {out_name}")

# -------------------- 入口 --------------------
def main():
    rootfile = sys.argv[1] if len(sys.argv) > 1 else "group_hists_out.root"
    if not os.path.isfile(rootfile):
        print(f("[ERROR] file not found: {rootfile}"))
        sys.exit(1)
    plot_all(rootfile)

if __name__ == "__main__":
    main()
