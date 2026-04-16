import os
import glob
import re
import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import pickle
import xgboost as xgb
import gc

from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from xgboost import XGBClassifier, plot_importance
from collections import defaultdict
from typing import Optional, Sequence, Tuple, List, Union

plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'
plt.style.use(hep.style.CMS)
# --- Configuration ---
BASE_DIR     = "/afs/ihep.ac.cn/users/y/yiyangzhao/Research/CMS_THU_Space/VVV/train/dataset"
SIG_DIR      = os.path.join(BASE_DIR, "signal")
BKG_DIR      = os.path.join(BASE_DIR, "bkg")
DATA_DIR     = os.path.join(BASE_DIR, "data")
RANDOM_STATE = 42
ENTRIES_PER_SAMPLE = 1000000
DECOR_LAMBDA = 30          # 去相关惩罚强度（如需更强平坦化可调大一些，建议 0.5~5 扫描）
_EPS = 1e-12                # 数值稳定项

global SCALE_2FAT
SCALE_2FAT = 0
global SCALE_3FAT
SCALE_3FAT = 0

global SCALE_2FAT_VVV
SCALE_2FAT_VVV = 0
global SCALE_2FAT_VH
SCALE_2FAT_VH = 0
global SCALE_2FAT_VV
SCALE_2FAT_VV = 0
global SCALE_2FAT_TT
SCALE_2FAT_TT = 0
global SCALE_2FAT_QCD
SCALE_2FAT_QCD = 0
global SCALE_3FAT_VVV
SCALE_3FAT_VVV = 0
global SCALE_3FAT_VH
SCALE_3FAT_VH = 0
global SCALE_3FAT_VV
SCALE_3FAT_VV = 0
global SCALE_3FAT_TT
SCALE_3FAT_TT = 0
global SCALE_3FAT_QCQ
SCALE_3FAT_QCD = 0

branches2 =  [
    "N_ak8", "N_ak4", "H_T",
    #"N_PV", "N_e", "N_mu", "N_gamma", 
    "pt8_1", "pt8_2", "eta8_1", "eta8_2", "msd8_1", "msd8_2", "mr8_1", "mr8_2",
    #"nConst8_1", "nConst8_2", "nCh8_1", "nCh8_2", "nEle8_1", "nEle8_2",
    #"nMu8_1", "nMu8_2", "nNh8_1", "nNh8_2", "nPho8_1", "nPho8_2",
    #"area8_1", "area8_2", "chEmEF8_1", "chEmEF8_2", "chHEF8_1", "chHEF8_2",
    #"hfEmEF8_1", "hfEmEF8_2", "hfHEF8_1", "hfHEF8_2", "neEmEF8_1", "neEmEF8_2",
    #"neHEF8_1", "neHEF8_2", "muEF8_1", "muEF8_2", "n2b1_1", "n2b1_2",
    #"n3b1_1", "n3b1_2", "tau21_1", "tau21_2", "tau32_1", "tau32_2",
    "WvsQCD_1", "WvsQCD_2",
    "pt4_1", "pt4_2", "pt4_3", "pt4_4", "eta4_1", "eta4_2", "eta4_3", "eta4_4",
    "mPF4_1", "mPF4_2", "mPF4_3", "mPF4_4", "nConst4_1", "nConst4_2",
    "nConst4_3", "nConst4_4", "nCh4_1", "nCh4_2", "nCh4_3", "nCh4_4",
    "nEle4_1", "nEle4_2", "nEle4_3", "nEle4_4",
    #"nMu4_1", "nMu4_2", "nMu4_3", "nMu4_4", 
    #"nNh4_1", "nNh4_2", "nNh4_3", "nNh4_4", "nPho4_1", "nPho4_2", "nPho4_3", "nPho4_4", 
    "area4_1", "area4_2", "area4_3", "area4_4", 
    "chEmEF4_1", "chEmEF4_2", "chEmEF4_3", "chEmEF4_4",
    "chHEF4_1", "chHEF4_2", "chHEF4_3", "chHEF4_4", 
    #"hfEmEF4_1", "hfEmEF4_2", "hfEmEF4_3", "hfEmEF4_4", "hfHEF4_1", "hfHEF4_2", "hfHEF4_3", "hfHEF4_4",
    "neEmEF4_1", "neEmEF4_2", "neEmEF4_3", "neEmEF4_4", "neHEF4_1", "neHEF4_2",
    "neHEF4_3", "neHEF4_4", 
    #"muEF4_1", "muEF4_2", "muEF4_3", "muEF4_4",
    "PT", "dR8", "dPhi", "m1overM", "m2overM","sphereM",
    #"M8", "M84", "pt1overPT", "pt2overPT", "PToverptsum", 
    "dR84_min", "dR44_min", "dR8L_min",
    "ptL_1", "ptL_2", "ptL_3", 
    "etaL_1", "etaL_2", "etaL_3", "phiL_1", "phiL_2", "phiL_3", 
    "isoEcalL_1", "isoEcalL_2", "isoEcalL_3", "isoHcalL_1", "isoHcalL_2", "isoHcalL_3"
]
branches3 = [
    "N_ak8", "N_ak4", "H_T",
    #"N_PV", "N_e", "N_mu", "N_gamma", 
    "pt8_1", "pt8_2", "pt8_3", "eta8_1", "eta8_2", "eta8_3",
    "msd8_1", "msd8_2", "msd8_3", "mr8_1", "mr8_2", "mr8_3",
    #"nConst8_1", "nConst8_2", "nConst8_3", "nCh8_1", "nCh8_2", "nCh8_3",
    #"nEle8_1", "nEle8_2", "nEle8_3", "nMu8_1", "nMu8_2", "nMu8_3",
    #"nNh8_1", "nNh8_2", "nNh8_3", "nPho8_1", "nPho8_2", "nPho8_3",
    #"area8_1", "area8_2", "area8_3", "chEmEF8_1", "chEmEF8_2", "chEmEF8_3",
    #"chHEF8_1", "chHEF8_2", "chHEF8_3", "hfEmEF8_1", "hfEmEF8_2", "hfEmEF8_3",
    #"hfHEF8_1", "hfHEF8_2", "hfHEF8_3", "neEmEF8_1", "neEmEF8_2", "neEmEF8_3",
    #"neHEF8_1", "neHEF8_2", "neHEF8_3", "muEF8_1", "muEF8_2", "muEF8_3",
    #"n2b1_1", "n2b1_2", "n2b1_3", "n3b1_1", "n3b1_2", "n3b1_3",
    #"tau21_1", "tau21_2", "tau21_3", "tau32_1", "tau32_2", "tau32_3",
    "WvsQCD_1", "WvsQCD_2", "WvsQCD_3",
    "sphereM", "M", "m1overM", "m2overM", "m3overM", 
    "PT",
    #"pt1overPT", "pt2overPT", "pt3overPT", "PToverptsum", 
    "dR_min", "dR_max", "dPhi_min", "dPhi_max", "dRL_min",
    "ptL_1", "ptL_2", "ptL_3", 
    "etaL_1", "etaL_2", "etaL_3", "phiL_1", "phiL_2", "phiL_3", 
    "isoEcalL_1", "isoEcalL_2", "isoEcalL_3",
    "isoHcalL_1", "isoHcalL_2", "isoHcalL_3"
]
BRANCH_CLIP_RANGES = {
    "H_T":   (0, 13600),
    "M":     (0, 13600),
    "M8":    (0, 13600),
    "M84":   (0, 13600),
    "PT":    (0, 13600),
    "PT":    (0, 13600),
    "pt4_1": (0, 13600),
    "pt4_2": (0, 13600),
    "pt4_3": (0, 13600),
    "pt4_4": (0, 13600),
    "pt8_1": (0, 13600),
    "pt8_2": (0, 13600),
    "pt8_3": (0, 13600),
    "ptL_1": (0, 13600),
    "ptL_2": (0, 13600),
    "ptL_3": (0, 13600)
}
# utility: group files by prefix without trailing _1, _2

def group_files(path):
    files = glob.glob(os.path.join(path, "*.root"))
    groups = {}
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        key  = re.sub(r'_[0-9]+$', '', name)
        if '202' in key:
            key = 'data'
        groups.setdefault(key, []).append(f)
    return groups
# x-section [pb] / raw entries * 1e8
_RAW_XSEC = {
    "www":              4.852,
    "wwz":              1.024,
    "wzz":              0.344,
    "zzz":              0.0863,
    "ww":               118.9,
    "wz":               45.16,
    "zz":               253.5,
    "qcd_ht100to200":   1.973e7,
    "qcd_ht200to400":   1.703e6,
    "qcd_ht400to600":   8.970e4,
    "qcd_ht600to800":   1.056e4,
    "qcd_ht800to1000":  2413,
    "qcd_ht1000to1200": 772.4,
    "qcd_ht1200to1500": 356.2,
    "qcd_ht1500to2000": 110.7,
    "qcd_ht2000toinf":  29.56,
    "tt_had":           79.80,
    "tt_semilep":       75.68,
    "zh":           2.26,
    "wplush":       3.72,
    "wminush":      2.15,
    "data":         20821
}

WEIGHT_2FAT = {}
WEIGHT_3FAT = {}

_RAW_XSEC_fat2 = {}
_RAW_XSEC_fat3 = {}
XSEC_MAP = {
    "fat2": _RAW_XSEC_fat2,
    "fat3": _RAW_XSEC_fat3,
}


RAW_XSEC = {
    k.lower().replace(' ', '_'): v
    for k, v in _RAW_XSEC.items()
}
def count_entries(path, tree_name):
    counts = {}
    groups = group_files(path)
    #print(groups)
    for sample, files in groups.items():
        total = 0
        for fpath in files:
            with uproot.open(fpath) as uf:
                if tree_name in uf:
                    tree = uf[tree_name]
                    total += tree.num_entries
        counts[sample] = total
    return counts

# 分别统计 signal/ bkg 下 fat2, fat3 的 entry
sig_fat2 = count_entries(SIG_DIR, "fat2")
sig_fat3 = count_entries(SIG_DIR, "fat3")
bkg_fat2 = count_entries(BKG_DIR, "fat2")
bkg_fat3 = count_entries(BKG_DIR, "fat3")
data_fat2 = count_entries(DATA_DIR, "fat2")
data_fat3 = count_entries(DATA_DIR, "fat3")

for sample, xsec in _RAW_XSEC.items():
    e2 = sig_fat2.get(sample, 0) + bkg_fat2.get(sample, 0) + data_fat2.get(sample, 0)
    e3 = sig_fat3.get(sample, 0) + bkg_fat3.get(sample, 0) + data_fat3.get(sample, 0)
    _RAW_XSEC_fat2[sample] = xsec * e2
    _RAW_XSEC_fat3[sample] = xsec * e3
    #print(f"{sample}: {e2} {e3}")
def filter_X(X: pd.DataFrame, y, w, branch: list, thresholds: dict = None, apply_to_sentinel: bool = True):
    """
    依据 thresholds 对指定列进行筛选，并处理哨兵值（< -990）。
    新增：thresholds 的每个键可支持“分多段的 & 或 |”的复合条件。

    条件 cond 的支持形式（对单列）：
      - 标量 c                 ->  col > c
      - 二元组 (mn, mx)        ->  (mn is None 或 col > mn) 且 (mx is None 或 col < mx)
      - 列表 [cond1, cond2...] ->  视作 OR：cond1 | cond2 | ...
      - 字典 {'|': [..]}       ->  多段 OR
      - 字典 {'&': [..]}       ->  多段 AND
    以上可递归嵌套，例如：{'&': [ (None,200), {'|':[ (0,40), (150,None) ]} ]}

    apply_to_sentinel=True 时：先剔除哨兵，再按数值条件筛选（若有条件）。
    apply_to_sentinel=False 时：哨兵行自动通过，其余行按数值条件筛选（若有条件）。
    """
    if thresholds is None:
        return X.copy(), y.copy(), w.copy()

    # 从全 True 开始，逐列收紧
    mask = pd.Series(True, index=X.index)

    def _combine_masks(masks: list[pd.Series], op: str, idx) -> pd.Series:
        if not masks:
            # 空集合在 AND 中返回全 True，在 OR 中返回全 False
            return pd.Series(True, index=idx) if op == '&' else pd.Series(False, index=idx)
        out = masks[0]
        for m in masks[1:]:
            out = (out & m) if op == '&' else (out | m)
        return out

    def _mask_from_cond(col: pd.Series, cond) -> pd.Series:
        """将任意支持的 cond 转成布尔掩码；不处理 sentinel。"""
        idx = col.index

        # None -> 全 True（相当于无条件）
        if cond is None:
            return pd.Series(True, index=idx)

        # 数值标量：col > c
        if isinstance(cond, (int, float, np.integer, np.floating)):
            return col > float(cond)

        # 二元组 (mn, mx)
        if isinstance(cond, tuple) and len(cond) == 2 and not isinstance(cond[0], (list, dict, tuple)):
            mn, mx = cond
            m = pd.Series(True, index=idx)
            if mn is not None:
                m &= (col > mn)
            if mx is not None:
                m &= (col < mx)
            return m

        # 列表：默认 OR
        if isinstance(cond, (list, tuple)):
            masks = [_mask_from_cond(col, c) for c in cond]
            return _combine_masks(masks, '|', idx)

        # 字典：{'|': [..]} 或 {'&': [..]}，支持递归
        if isinstance(cond, dict):
            if '|' in cond:
                masks = [_mask_from_cond(col, c) for c in cond['|']]
                return _combine_masks(masks, '|', idx)
            if '&' in cond:
                masks = [_mask_from_cond(col, c) for c in cond['&']]
                return _combine_masks(masks, '&', idx)
            # 兼容小写字符串键
            if 'or' in cond:
                masks = [_mask_from_cond(col, c) for c in cond['or']]
                return _combine_masks(masks, '|', idx)
            if 'and' in cond:
                masks = [_mask_from_cond(col, c) for c in cond['and']]
                return _combine_masks(masks, '&', idx)
            raise ValueError(f"Unsupported dict condition keys for column: {cond}")

        raise TypeError(f"Unsupported condition type for column: {type(cond)}")

    for b in branch:
        if b not in X.columns:
            raise KeyError(f"Column {b!r} not found in X")

        col = X[b]
        cond = thresholds.get(b, None)
        sentinel = col < -990  # True 表示哨兵值行

        if apply_to_sentinel:
            # 先剔除所有哨兵行
            mask &= ~sentinel
            # 再对剩下的行应用数值阈值
            if cond is not None:
                cond_mask = _mask_from_cond(col, cond)
                mask &= cond_mask
        else:
            # 哨兵行自动通过，只有非哨兵才做阈值判断
            if cond is not None:
                cond_mask = _mask_from_cond(col, cond)
                mask &= (cond_mask | sentinel)

    return X.loc[mask].copy(), y[mask.values].copy(), w[mask.values].copy()

def standardize_X(X: pd.DataFrame) -> pd.DataFrame:
    # 需要做 ln 变换（b）的列名规则（区分大小写）
    def _need_ln(col: str) -> bool:
        if col.startswith("N_"):
            return True
        if col == "H_T":
            return True
        if col.startswith("pt"):
            return True
        if col.startswith("sphereM"):
            return True
        if col.startswith("iso"):
            return True
        if col.startswith("M"):
            return True
        if col == "PT":
            return True
        if col.startswith("mPF"):
            return True
        if col.startswith(("nMu", "nNh", "nPho", "nCh", "nEle", "nConst")):
            return True
        if "overPT" in col:
            return True
        if col.startswith("mr"):
            return True
        return False

    for col in X.columns:
        arr = X[col].values  # 底层 numpy 视图（通常是 float；若不是，下面会处理）

        # 1) 哨兵值：-999（这里沿用原逻辑：< -990 都视为哨兵），不改变
        mask = arr < -990
        valid = ~mask
        if not valid.any():
            continue

        # ---- 2) 进行裁切（仅对非哨兵值） --------------------------------------
        clip_min, clip_max = BRANCH_CLIP_RANGES.get(col, (None, None))
        if clip_min is not None:
            arr[valid & (arr < clip_min)] = clip_min
        if clip_max is not None:
            arr[valid & (arr > clip_max)] = clip_max

        # ---- 3) 不再做均值方差标准化：只按规则选择是否做 ln 变换 --------------
        if _need_ln(col):
            # 为了避免 ln(0) -> -inf，这里只对 >0 的有效值取 ln；<=0 的有效值保持不变
            pos = valid & (arr > 0)
            if pos.any():
                # 若列是整型，这里会产生浮点结果；确保写回不会被截断
                if not np.issubdtype(arr.dtype, np.floating):
                    arr = arr.astype(float, copy=True)
                arr[pos] = np.log(arr[pos])

        X[col] = arr

    return X

# Multi-class
TYPE = {
    "www":              0,
    "wwz":              1,
    "wzz":              2,
    "zzz":              3,
    "zh":               4,
    "wplush":           5,
    "wminush":          5,
    "qcd_ht100to200":   10,
    "qcd_ht200to400":   10,
    "qcd_ht400to600":   10,
    "qcd_ht600to800":   10,
    "qcd_ht800to1000":  10,
    "qcd_ht1000to1200": 10,
    "qcd_ht1200to1500": 10,
    "qcd_ht1500to2000": 10,
    "qcd_ht2000toinf":  10,
    "tt_had":           11,
    "tt_semilep":       12,
    "ww":               13,
    "wz":               14,
    "zz":               15,
    "data":             -2,
    "unknown":          -1,
}
def _map_to_3classes(y_raw: np.ndarray) -> np.ndarray:
    y_raw = np.asarray(y_raw)
    y = np.full_like(y_raw, fill_value=-999, dtype=int)

    # VVV 类
    vvv_codes = {TYPE["www"], TYPE["wwz"], TYPE["wzz"], TYPE["zzz"]}
    # VH  类
    vh_codes  = {TYPE["zh"], TYPE["wplush"], TYPE["wminush"]}
    # TTbar 类
    tt_codes  = {TYPE["tt_had"], TYPE["tt_semilep"]}
    # VV 类
    vv_codes  = {TYPE["ww"], TYPE["wz"], TYPE["zz"]}
    # QCD 类
    qcd_codes = {TYPE["qcd_ht100to200"], TYPE["qcd_ht200to400"], TYPE["qcd_ht400to600"], TYPE["qcd_ht600to800"], TYPE["qcd_ht800to1000"], TYPE["qcd_ht1000to1200"], TYPE["qcd_ht1200to1500"], TYPE["qcd_ht1500to2000"], TYPE["qcd_ht2000toinf"]}
    # 排除
    exclude   = {TYPE["data"], TYPE["unknown"]}

    # 标注
    y[np.isin(y_raw, list(vvv_codes))] = 0
    y[np.isin(y_raw, list(vh_codes))]  = 1
    y[np.isin(y_raw, list(tt_codes))] = 2
    y[np.isin(y_raw, list(vv_codes))]  = 3
    y[np.isin(y_raw, list(qcd_codes))]  = 4
    # 背景（不是排除，并且不是上面两类的，均归为 BKG=0）
    mask_valid_not_set = (~np.isin(y_raw, list(exclude))) & (y == -999)
    y[mask_valid_not_set] = -999

    return y, ~np.isin(y_raw, list(exclude))

def _smooth_boost(prob: np.ndarray, floor: float, bmin: float, bmax, power: float) -> np.ndarray:
    prob = np.asarray(prob)
    bmax_arr = np.asarray(bmax)
    eps = 1e-8
    denom = np.maximum(1.0 - floor, eps)
    t = np.clip((prob - floor) / denom, 0.0, 1.0) ** power
    return bmin + (bmax_arr - bmin) * t
def prepare_multi_data(tree_name, branches, entries_per_sample,
                 SIG_DIR=SIG_DIR, BKG_DIR=BKG_DIR):

    def load_samples(path):
        """
        按 sample 分组读取文件，每个 sample 最多取 limit_per_sample 条目，
        分配均匀的 per-event weight，使得 sum(weight_i)=xsec(sample)。
        """
        groups = group_files(path)  # { sample_id: [file1.root, file2.root, ...] }
        dfs = []
        
        for sample_id, file_list in groups.items():
            sid = sample_id.lower()
            limit = entries_per_sample
            if 'data' in sample_id:
            #    #sample_id = 'data'
            #    sid = 'data'
                limit = 1e3
            #    #print(limit)
            if sid not in XSEC_MAP[tree_name]:
                raise KeyError(f"Sample '{sample_id}' not found in RAW_XSEC map")
            xsec = XSEC_MAP[tree_name][sample_id]
            
            parts = []
            n_read = 0
            for fpath in file_list:
                remain = limit - n_read
                if remain <= 0:
                    break

                with uproot.open(fpath) as uf:
                    tree = uf[tree_name]
                    # 本文件总条目数
                    n_entries = tree.num_entries
                    # 本次从该文件中要读取的条目数
                    n_to_read = min(remain, n_entries)
                    if n_to_read <= 0:
                        continue

                    # 只读取前 n_to_read 条
                    df_part = tree.arrays(
                        branches + ["type"],
                        library='pd',
                        entry_start=0,
                        entry_stop=n_to_read
                    )

                parts.append(df_part)
                n_read += len(df_part)
                if n_read >= limit:
                    break
            
            if n_read == 0:
                continue
            
            df_sample = pd.concat(parts, ignore_index=True)
            del parts
            gc.collect()
            # 2. 均匀分配 weight 使 sum_i w_i == xsec
            per_evt_w = xsec / float(n_read)
            df_sample["weight"]   = per_evt_w

            df_sample["type"] = int(TYPE[sample_id])
            type = df_sample["type"].values[0]
            
            # 4. 每个 sample 内部打乱、重置索引
            #df_sample = df_sample.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
            dfs.append(df_sample)
            print(f"Sample '{sample_id}': read {n_read} entries (limit {limit}) with weight {per_evt_w} Type: {type}")
            if (tree_name == "fat2"):
                WEIGHT_2FAT[sample_id] = per_evt_w
            elif (tree_name == "fat3"):
                WEIGHT_3FAT[sample_id] = per_evt_w
        
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        else:
            # 万一某类全没读到，返回空 DataFrame
            cols = branches + ["weight", "type"]
            return pd.DataFrame(columns=cols)
    
    # 读取 signal、background
    df_sig = load_samples(SIG_DIR)
    df_bkg = load_samples(BKG_DIR)
    df_data = load_samples(DATA_DIR)

    # 3. 缩放总权重 = 背景总权重，对于TYPE中的VVV、VH分别计算scale
    w_sig_vvv = df_sig[df_sig["type"].isin([TYPE["www"], TYPE["wwz"], TYPE["wzz"], TYPE["zzz"]])]["weight"].sum()
    w_sig_vh  = df_sig[df_sig["type"].isin([TYPE["zh"], TYPE["wplush"], TYPE["wminush"]])]["weight"].sum()
    w_bkg_vv  = df_bkg[df_bkg["type"].isin([TYPE["ww"], TYPE["wz"], TYPE["zz"]])]["weight"].sum()
    w_bkg_tt  = df_bkg[df_bkg["type"].isin([TYPE["tt_had"], TYPE["tt_semilep"]])]["weight"].sum()
    w_bkg_qcd = df_bkg[df_bkg["type"].isin([TYPE["qcd_ht100to200"], TYPE["qcd_ht200to400"], TYPE["qcd_ht400to600"], TYPE["qcd_ht600to800"], TYPE["qcd_ht800to1000"], TYPE["qcd_ht1000to1200"], TYPE["qcd_ht1200to1500"], TYPE["qcd_ht1500to2000"], TYPE["qcd_ht2000toinf"]])]["weight"].sum()
    
    n_sig_vvv = len(df_sig[df_sig["type"].isin([TYPE["www"], TYPE["wwz"], TYPE["wzz"], TYPE["zzz"]])])
    n_sig_vh  = len(df_sig[df_sig["type"].isin([TYPE["zh"], TYPE["wplush"], TYPE["wminush"]])])
    n_bkg_vv  = len(df_bkg[df_bkg["type"].isin([TYPE["ww"], TYPE["wz"], TYPE["zz"]])])
    n_bkg_tt  = len(df_bkg[df_bkg["type"].isin([TYPE["tt_had"], TYPE["tt_semilep"]])])
    n_bkg_qcd = len(df_bkg[df_bkg["type"].isin([TYPE["qcd_ht100to200"], TYPE["qcd_ht200to400"], TYPE["qcd_ht400to600"], TYPE["qcd_ht600to800"], TYPE["qcd_ht800to1000"], TYPE["qcd_ht1000to1200"], TYPE["qcd_ht1200to1500"], TYPE["qcd_ht1500to2000"], TYPE["qcd_ht2000toinf"]])])
    
    if w_sig_vvv > 0 and w_sig_vh > 0:
        scale_vvv  = 1e10 / (w_sig_vvv)
        scale_vh   = 1e10 / (w_sig_vh)
        scale_vv   = 1e10 / (w_bkg_vv)
        scale_tt   = 1e10 / (w_bkg_tt)
        scale_qcd  = 1e10 / (w_bkg_qcd)
        
        df_sig.loc[df_sig["type"].isin([TYPE["www"], TYPE["wwz"], TYPE["wzz"], TYPE["zzz"]]), "weight"] = df_sig.loc[df_sig["type"].isin([TYPE["www"], TYPE["wwz"], TYPE["wzz"], TYPE["zzz"]]), "weight"] * scale_vvv
        df_sig.loc[df_sig["type"].isin([TYPE["zh"], TYPE["wplush"], TYPE["wminush"]]), "weight"] = df_sig.loc[df_sig["type"].isin([TYPE["zh"], TYPE["wplush"], TYPE["wminush"]]), "weight"] * scale_vh
        df_bkg.loc[df_bkg["type"].isin([TYPE["ww"], TYPE["wz"], TYPE["zz"]]), "weight"] = df_bkg.loc[df_bkg["type"].isin([TYPE["ww"], TYPE["wz"], TYPE["zz"]]), "weight"] * scale_vv
        df_bkg.loc[df_bkg["type"].isin([TYPE["tt_had"], TYPE["tt_semilep"]]), "weight"] = df_bkg.loc[df_bkg["type"].isin([TYPE["tt_had"], TYPE["tt_semilep"]]), "weight"] * scale_tt
        df_bkg.loc[df_bkg["type"].isin([TYPE["qcd_ht100to200"], TYPE["qcd_ht200to400"], TYPE["qcd_ht400to600"], TYPE["qcd_ht600to800"], TYPE["qcd_ht800to1000"], TYPE["qcd_ht1000to1200"], TYPE["qcd_ht1200to1500"], TYPE["qcd_ht1500to2000"], TYPE["qcd_ht2000toinf"]]), "weight"] = df_bkg.loc[df_bkg["type"].isin([TYPE["qcd_ht100to200"], TYPE["qcd_ht200to400"], TYPE["qcd_ht400to600"], TYPE["qcd_ht600to800"], TYPE["qcd_ht800to1000"], TYPE["qcd_ht1000to1200"], TYPE["qcd_ht1200to1500"], TYPE["qcd_ht1500to2000"], TYPE["qcd_ht2000toinf"]]), "weight"] * scale_qcd

        if tree_name == "fat2": 
            SCALE_2FAT_VVV = scale_vvv
            SCALE_2FAT_VH  = scale_vh
            SCALE_2FAT_VV = scale_vv
            SCALE_2FAT_TT = scale_tt
            SCALE_2FAT_QCD = scale_qcd
            print(f"SCALE_2FAT_VVV: {scale_vvv}")
            print(f"SCALE_2FAT_VH: {scale_vh}")
            print(f"SCALE_2FAT_VV: {scale_vv}")
            print(f"SCALE_2FAT_TT: {scale_tt}")
            print(f"SCALE_2FAT_QCD: {scale_qcd}")
        elif tree_name == "fat3": 
            SCALE_3FAT_VVV = scale_vvv
            SCALE_3FAT_VH  = scale_vh
            SCALE_3FAT_VV = scale_vv
            SCALE_3FAT_TT = scale_tt
            SCALE_3FAT_QCD = scale_qcd
            print(f"SCALE_3FAT_VVV: {scale_vvv}")
            print(f"SCALE_3FAT_VH: {scale_vh}")
            print(f"SCALE_3FAT_VV: {scale_vv}")
            print(f"SCALE_3FAT_TT: {scale_tt}")
            print(f"SCALE_3FAT_QCD: {scale_qcd}")

    # 4. 合并后全体再打乱、重置索引
    df_all = pd.concat([df_sig, df_bkg], ignore_index=True)
    del df_sig, df_bkg
    gc.collect()
    
    df_all = df_all.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    # 5. 拆分标签、权重、特征
    X_data = df_data[branches]
    y_data = df_data.pop("type").values
    w_data = df_data.pop("weight").values
    del df_data
    gc.collect()
    
    X = df_all[branches]
    y = df_all.pop("type").values
    w = df_all.pop("weight").values
    
    del df_all
    gc.collect()

    return X, y, w, X_data, y_data, w_data

def _cvm_flatness_value(
    y: np.ndarray,
    Z: np.ndarray,
    w: np.ndarray,
    n_bins: int = 10,
    power: float = 2.0,
) -> float:
    """
    计算 Cramer-von Mises 型 flatness 惩罚的数值（和 _cvm_flatness_neg_grad_wrt_y 对应）：

      L_flat ~ sum_{decor j} sum_{z-bin b} sum_{i in bin b}
                 w_abs_i * |F_b(y_i) - F_global(y_i)|^power

    其中 F_b, F_global 为加权经验 CDF，权重使用 |w|（忽略符号），
    确保 flat_penalty 始终 >= 0。

    参数：
      y      : raw logit，shape (n,)
      Z      : 去相关变量矩阵，shape (n, n_decor)
      w      : 原始样本权重（允许有负权重），shape (n,)
      n_bins : z 的分箱数
      power  : 幂指数，通常取 2

    返回：
      flat_penalty : 一个非负标量，越小表示越“平坦”（越独立）
    """
    y = np.asarray(y, dtype=float).ravel()
    Z = np.asarray(Z, dtype=float)
    w = np.asarray(w, dtype=float).ravel()

    n = y.shape[0]
    if n == 0 or Z.size == 0:
        return 0.0

    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    _, n_decor = Z.shape

    # 使用绝对值权重来度量“统计量”，避免负权重导致 flat_penalty < 0
    W_abs = float(np.sum(w) + _EPS)
    if W_abs <= _EPS:
        return 0.0

    # 归一化权重，用于 CDF 计算
    w_norm = w / W_abs

    # 全局经验 CDF 位置 F_global(y)
    global_pos = _weighted_ecdf_positions(y, w_norm)

    flat_penalty = 0.0
    for j in range(n_decor):
        zj = Z[:, j]
        z_min = float(np.min(zj))
        z_max = float(np.max(zj))
        if (not np.isfinite(z_min)) or (not np.isfinite(z_max)) or (z_min == z_max):
            # 这个 decor 变量无有效跨度，跳过
            continue

        edges = np.linspace(z_min, z_max, n_bins + 1, dtype=float)
        bin_idx = np.searchsorted(edges, zj, side="right") - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)

        for b in range(n_bins):
            idx_b = np.nonzero(bin_idx == b)[0]
            # 样本太少的 bin 不做估计，避免统计涨落过大
            if idx_b.size < 3:
                continue

            local_pos = _weighted_ecdf_positions(y[idx_b], w_norm[idx_b])
            diff = local_pos - global_pos[idx_b]

            # power=2 时为平方差；w_norm>=0，保证 flat_penalty >= 0
            flat_penalty += float(np.sum(w_norm[idx_b] * (np.abs(diff) ** power)))

    return float(flat_penalty)


def _weighted_ecdf_positions(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    加权经验分布的位置：对每个样本 i，返回
      pos_i = (累计权重 <= y_i) / sum_j w_j
    实现方式与 hep_ml.losses._compute_positions 相同：
      先按 y 排序，再用归一化权重的累积和构造位置。

    参数：
      y : 一维数组，长度 n
      w : 对应的加权，长度 n（可以是原始权重或归一化权重）

    返回：
      pos : shape (n,) 的位置数组，取值约在 (0, 1) 内
    """
    y = np.asarray(y, dtype=float).ravel()
    w = np.asarray(w, dtype=float).ravel()
    n = y.shape[0]
    if n == 0:
        return np.zeros_like(y)

    # 按 y 排序
    order = np.argsort(y)
    w_sorted = w[order].astype(float)
    # 归一化权重
    W = float(np.sum(w_sorted)) + _EPS
    w_sorted /= W

    # 中点位置：累计权重减去一半本身，避免所有点都挤在边界
    cum = np.cumsum(w_sorted) - 0.5 * w_sorted
    pos_sorted = cum

    # 还原到原始顺序
    pos = np.empty_like(pos_sorted)
    pos[order] = pos_sorted
    return pos


def _cvm_flatness_neg_grad_wrt_y(
    y: np.ndarray,
    groups: Sequence[np.ndarray],
    w: np.ndarray,
    power: float = 2.0
) -> np.ndarray:
    """
    Cramer-von Mises 风格 flatness 惩罚的“负梯度”（近似），
    与 hep_ml 中 BinFlatnessLossFunction 的思想一致：

      L_flat ~ sum_groups sum_{i in group} w_abs_i * |F_group(y_i) - F_global(y_i)|^power

    其中 F_group, F_global 为加权经验 CDF。
    这里对 decor 项使用权重的绝对值 |w|，确保每个事件在 decor 上的贡献为“强度”，
    而不是受 NLO 符号影响。

    参数：
      y      : raw logit，shape (n,)
      groups : 一组 index 数组，每个数组是一段 z-bin 的样本索引
      w      : 原始样本权重（允许有负权重），shape (n,)
      power  : 幂指数，通常取 2；power=2 时是标准的 CvM 型损失

    返回：
      neg_grad : 近似 -dL_flat / dy_i，shape (n,)
    """
    y = np.asarray(y, dtype=float).ravel()
    w = np.asarray(w, dtype=float).ravel()
    n = y.shape[0]
    if n == 0:
        return np.zeros_like(y)
    if groups is None or len(groups) == 0:
        return np.zeros_like(y)

    # 使用绝对值权重来定义 decor 的统计量
    W_abs = float(np.sum(w) + _EPS)
    if W_abs <= _EPS:
        return np.zeros_like(y)

    # 归一化权重用于计算 CDF 位置
    w_norm = w / W_abs

    # 全局位置 F_global(y)
    global_pos = _weighted_ecdf_positions(y, w_norm)

    neg_grad = np.zeros_like(y)
    for idx in groups:
        idx = np.asarray(idx, dtype=int)
        if idx.size < 2:
            continue

        # 本 group 内的局部 CDF 位置 F_group(y_i)
        local_pos = _weighted_ecdf_positions(y[idx], w_norm[idx])
        glob = global_pos[idx]
        diff = local_pos - glob

        # power=2 时 bin_grad = 2 * (local - global)
        bin_grad = power * np.sign(diff) * (np.abs(diff) ** (power - 1.0))
        neg_grad[idx] += bin_grad

    # decor 梯度按 |w| 的归一化权重缩放
    neg_grad *= w_norm
    return neg_grad

def check_weights(w, name="w"):
    w = np.asarray(w, dtype=float).ravel()

    finite_mask = np.isfinite(w)
    if not np.all(finite_mask):
        bad = np.where(~finite_mask)[0]
        print(f"[{name}] 非有限值数量: {bad.size} (nan/inf). 例如索引: {bad[:10].tolist()}")
    else:
        print(f"[{name}] 全部有限 (no nan/inf)")

    n = w.size
    n_pos = int(np.sum(w > 0))
    n_zero = int(np.sum(w == 0))
    n_neg = int(np.sum(w < 0))

    print(f"[{name}] N={n}")
    print(f"[{name}] >0: {n_pos}  ({n_pos/n:.6f})")
    print(f"[{name}] =0: {n_zero} ({n_zero/n:.6f})")
    print(f"[{name}] <0: {n_neg}  ({n_neg/n:.6f})")
    print(f"[{name}] min={np.nanmin(w):.6g}, max={np.nanmax(w):.6g}")
    print(f"[{name}] sum={np.nansum(w):.6g}, sum_abs={np.nansum(np.abs(w)):.6g}")

    if n_neg > 0:
        idx = np.where(w < 0)[0]
        print(f"[{name}] 负权重示例: idx={idx[:10].tolist()}, values={w[idx[:10]]}")


def _resolve_decor_indices(
    X,
    decorrelate_feature_names: Optional[Sequence[Union[str, int]]]
) -> List[int]:
    """
    把用户给的 '需要去相关' 的变量名（或列号）解析为列索引列表。
    支持：
      - 若 X 为 pandas.DataFrame，则直接按列名解析；
      - 否则，若全局定义了 FEATURE_NAMES（list[str]），则从中解析；
      - 否则，若给的是 int（列号），则直接使用；
      - 否则报错。
    """
    if not decorrelate_feature_names:
        return []

    # 1) pandas.DataFrame 情况
    try:
        if isinstance(X, pd.DataFrame):
            name_to_idx = {c: i for i, c in enumerate(list(X.columns))}
            idx = []
            for key in decorrelate_feature_names:
                if isinstance(key, int):
                    idx.append(int(key))
                else:
                    if key not in name_to_idx:
                        raise ValueError(f"去相关变量名 '{key}' 不在 DataFrame 列中。")
                    idx.append(name_to_idx[key])
            return sorted(set(idx))
    except Exception:
        pass

    # 3) 纯 numpy，且用户传了列号
    idx = []
    for key in decorrelate_feature_names:
        if isinstance(key, int):
            idx.append(int(key))
        else:
            raise ValueError(
                "无法从名字解析列索引：X 不是 DataFrame，且未提供全局 FEATURE_NAMES。"
                "你可以：a) 传入 DataFrame；b) 定义全局 FEATURE_NAMES；或 c) 直接传列号(int)。"
            )
    return sorted(set(idx))


def _weighted_stats_centered(y: np.ndarray, z: np.ndarray, w_norm: np.ndarray):
    """
    计算加权均值/中心化/方差/协方差等（基于归一化权重 w_norm, sum=1）
    返回：y0, z0, var_y, var_z, cov, sig_y, sig_z, corr
    """
    y_mean = np.sum(w_norm * y)
    z_mean = np.sum(w_norm * z)
    y0 = y - y_mean
    z0 = z - z_mean
    var_y = np.sum(w_norm * y0 * y0) + _EPS
    var_z = np.sum(w_norm * z0 * z0) + _EPS
    sig_y = np.sqrt(var_y)
    sig_z = np.sqrt(var_z)
    cov = np.sum(w_norm * y0 * z0)
    corr = cov / (sig_y * sig_z + _EPS)
    return y0, z0, var_y, var_z, cov, sig_y, sig_z, corr


def _make_multiclass_objective_with_decor(
    num_class: int,
    Z_train: np.ndarray,      # shape (n_samples, n_decor)
    w_train: np.ndarray,      # shape (n_samples,)
    lam: float,
):
    """
    多类 logloss + CvM flatness 惩罚。

    统一约定：
      - 类内：用原始 w 做加权平均；
      - 类间：各类的“平均损失”直接相加。

    分类项：
      L_cls = sum_k [ sum_{i: y_i=k} w_i * ell_i / sum_{i: y_i=k} w_i ]

    decor 项：
      对每个类 k：
        使用 w_cls_k（只在类 k 非零）计算 CvM L_flat^k，
      总 decor loss = lam * sum_k L_flat^k。

    重要修复：
      1) 由于“类内平均”会让每个 class 的有效权重总和=1，导致 Hessian 总和过小，
         若 min_child_weight>=1 会直接无法生长任何树，训练表现为 loss 完全不变。
         这里对 *整个 objective* 乘一个常数 scale = n_train / num_class，仅改变整体尺度，不改变类间/类内相对权重结构。
      2) XGBoost>=2.1 要求自定义 objective 返回 grad/hess 形状为 (n_samples, n_classes)。
      3) 动态 LR：前 400 轮等效 0.01；第 400-499 轮等效 0.005；之后保持 0.005。
         为保证前 400 轮“完全不变”，用 lr_mult = lr(t)/0.01，只在 grad 上乘该系数。
    """
    Z_train = np.asarray(Z_train, dtype=float)
    w_train = np.asarray(w_train, dtype=float).ravel()
    n_train = w_train.shape[0]

    # === 预先用 Z_train 构造 decor 的分组（每个 decor 变量一个 group 列表） ===
    if Z_train.ndim == 1:
        Z_train = Z_train.reshape(-1, 1)
    n_samples, n_decor = Z_train.shape

    N_DECOR_BINS = 5
    decor_groups_list: List = []
    if lam > 0.0 and n_decor > 0:
        for j in range(n_decor):
            zj = Z_train[:, j]
            z_min = float(np.min(zj))
            z_max = float(np.max(zj))
            if (not np.isfinite(z_min)) or (not np.isfinite(z_max)) or (z_min == z_max):
                decor_groups_list.append(None)
                continue

            edges = np.linspace(z_min, z_max, N_DECOR_BINS + 1, dtype=float)
            bin_idx = np.searchsorted(edges, zj, side="right") - 1
            bin_idx = np.clip(bin_idx, 0, N_DECOR_BINS - 1)

            groups_j: List[np.ndarray] = []
            for b in range(N_DECOR_BINS):
                idx_b = np.nonzero(bin_idx == b)[0]
                if idx_b.size >= 3:
                    groups_j.append(idx_b)

            decor_groups_list.append(groups_j if len(groups_j) > 0 else None)
    else:
        decor_groups_list = [None] * n_decor

    # 关键：整体尺度修复（只影响 scale，不改变你定义的“类内平均/类间相加”的相对结构）
    _scale = float(n_train) / float(num_class) if n_train > 0 else 1.0

    # === 动态 LR 状态（用 objective 被调用次数近似 boosting 轮数；通常每轮训练会调用一次）===
    _base_lr = 0.05
    _iter_state = {"t": 0}

    def obj(y_true: np.ndarray, y_pred_in: np.ndarray):
        """
        sklearn.XGBClassifier 自定义 objective 的签名：
          y_true: shape (n,)
          y_pred_in: raw margin，可能是 (n*num_class,) 或 (n, num_class)

        返回：
          grad, hess: shape (n, num_class)
        """
        # ---- dynamic LR multiplier ----
        t = int(_iter_state["t"])
        _iter_state["t"] = t + 1

        if t < 1000:
            lr_t = 0.01
        elif t < 1200:
            lr_t = 0.01 - (0.01 - 0.005) * (t - 1000) / 200
        else:
            lr_t = 0.005
        lr_mult = float(lr_t) / float(_base_lr)  # 1.0 或 0.5

        y_true = np.asarray(y_true, dtype=int)
        n = y_true.shape[0]
        assert n == n_train, f"训练集大小不一致: n={n}, n_train={n_train}"

        y_pred_in = np.asarray(y_pred_in, dtype=float)
        if y_pred_in.ndim == 1:
            y_pred = y_pred_in.reshape(n, num_class)
        elif y_pred_in.ndim == 2:
            if y_pred_in.shape == (n, num_class):
                y_pred = y_pred_in
            elif y_pred_in.shape == (num_class, n):
                y_pred = y_pred_in.T
            else:
                y_pred = y_pred_in.reshape(n, num_class)
        else:
            y_pred = y_pred_in.reshape(n, num_class)

        # ---------- softmax 概率 ----------
        y_shift = y_pred - np.max(y_pred, axis=1, keepdims=True)
        exp_y = np.exp(y_shift)
        P = exp_y / (np.sum(exp_y, axis=1, keepdims=True) + _EPS)

        y_true_int = y_true
        w_used = w_train  # 与训练集顺序完全对齐

        # ---------- 1) 分类项：类内加权平均，类间直接相加 ----------
        class_w_sum = np.bincount(
            y_true_int,
            weights=w_used,
            minlength=num_class
        ).astype(float)
        class_w_sum[class_w_sum <= _EPS] = _EPS

        # 有效权重：w_eff_i = (w_i / S_{y_i}) * scale
        w_eff = (w_used / class_w_sum[y_true_int]) * _scale

        grad_cls = P.copy()
        grad_cls[np.arange(n), y_true_int] -= 1.0
        grad_cls *= w_eff[:, None]

        hess_cls = (P * (1.0 - P)) * w_eff[:, None]

        # ---------- 2) decor 梯度：每个类的 CvM 梯度，再乘 lam 相加 ----------
        grad_dec = np.zeros_like(grad_cls)
        if lam > 0.0 and n_decor > 0:
            for k in range(num_class):
                mask_k = (y_true_int == k)
                if not np.any(mask_k):
                    continue

                yk = y_pred[:, k]

                w_cls_k = np.zeros_like(w_used)
                w_cls_k[mask_k] = w_used[mask_k]

                gk = np.zeros(n, dtype=float)
                for j in range(n_decor):
                    groups_j = decor_groups_list[j]
                    if groups_j is None:
                        continue
                    neg_grad_flat = _cvm_flatness_neg_grad_wrt_y(
                        yk,
                        groups_j,
                        w_cls_k,
                        power=2.0
                    )
                    gk += (-neg_grad_flat)

                # 与分类项同整体尺度一致
                grad_dec[:, k] = (lam * _scale) * gk

        # ---------- 3) 合并梯度 / Hessian ----------
        grad = grad_cls + grad_dec

        # 动态 LR：只缩放梯度（前 400 轮 lr_mult=1，不改变；之后 lr_mult=0.5）
        grad *= lr_mult

        hess = hess_cls + 1e-6

        grad = np.asarray(grad, dtype=np.float32)
        hess = np.asarray(hess, dtype=np.float32)
        return grad, hess

    return obj



def _make_multiclass_total_loss_metric(
    num_class: int,
    Z_train: np.ndarray,
    Z_test: np.ndarray,
    w_train: np.ndarray,
    w_test: np.ndarray,
    lam: float,
):
    """
    自定义 eval_metric，对应的总损失：

      total_loss = L_cls + lam * sum_k L_flat^{(k)}

    其中：
      1) L_cls：类内加权平均、类间直接相加
      2) decor：对每个类用 w_cls_k 计算 flatness，再类间相加乘 lam

    这里不做 objective 的 scale 修复（metric 保持物理含义与可读性）。
    """
    Z_train = np.asarray(Z_train, dtype=float)
    Z_test  = np.asarray(Z_test,  dtype=float)
    w_train = np.asarray(w_train, dtype=float).ravel()
    w_test  = np.asarray(w_test,  dtype=float).ravel()

    n_train = w_train.shape[0]
    n_test  = w_test.shape[0]

    N_DECOR_BINS = 5

    def feval(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        sklearn.XGBClassifier 自定义 eval_metric 签名：
          - y_true: shape (n,)
          - y_pred: shape (n * num_class,) 或 (n, num_class)
                   注意：在某些版本/配置下也可能已经是概率。

        返回一个标量 total_loss，日志里显示为 'feval'。
        """
        y_true = np.asarray(y_true, dtype=int)
        n = y_true.shape[0]

        # 用长度判断是在 train 还是 test 上
        if n == n_train:
            Z = Z_train[:n, :]
            w = w_train[:n]
        elif n == n_test:
            Z = Z_test[:n, :]
            w = w_test[:n]
        else:
            Z = Z_train[:n, :]
            w = w_train[:n]

        y_pred = np.asarray(y_pred, dtype=float)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(n, num_class)
        elif y_pred.ndim == 2:
            if y_pred.shape == (n, num_class):
                pass
            elif y_pred.shape == (num_class, n):
                y_pred = y_pred.T
            else:
                y_pred = y_pred.reshape(n, num_class)
        else:
            y_pred = y_pred.reshape(n, num_class)

        # --- 判断 y_pred 是否已经是概率（稳妥兼容） ---
        # 若每行和≈1 且都在 [0,1]，则认为是 prob；否则认为是 raw margin 需要 softmax
        row_sum = np.sum(y_pred, axis=1)
        is_prob = (
            np.all(np.isfinite(y_pred)) and
            np.all(y_pred >= -1e-6) and np.all(y_pred <= 1.0 + 1e-6) and
            np.all(np.isfinite(row_sum)) and
            np.mean(np.abs(row_sum - 1.0)) < 1e-3
        )

        if is_prob:
            P = np.clip(y_pred, _EPS, 1.0)
            P = P / (np.sum(P, axis=1, keepdims=True) + _EPS)
        else:
            y_shift = y_pred - np.max(y_pred, axis=1, keepdims=True)
            exp_y = np.exp(y_shift)
            P = exp_y / (np.sum(exp_y, axis=1, keepdims=True) + _EPS)

        y_true_int = y_true
        w = np.asarray(w, dtype=float).ravel()

        # ---------- 1) 分类 logloss：类内加权平均，类间直接相加 ----------
        class_w_sum = np.bincount(
            y_true_int,
            weights=w,
            minlength=num_class
        ).astype(float)
        class_w_sum[class_w_sum <= _EPS] = _EPS

        w_eff = w / class_w_sum[y_true_int]

        p_true = P[np.arange(n), y_true_int]
        ell = -np.log(p_true + _EPS)
        logloss = float(np.sum(w_eff * ell))

        # ---------- 2) decor：lam * sum_k L_flat^{(k)} ----------
        flat_penalty = 0.0
        if lam > 0.0 and Z.size > 0:
            for k in range(num_class):
                mask_k = (y_true_int == k)
                if not np.any(mask_k):
                    continue

                w_cls_k = np.zeros_like(w)
                w_cls_k[mask_k] = w[mask_k]

                yk = y_pred[:, k] if not is_prob else np.log(P[:, k] + _EPS)  # prob 情况下用 logit-like 也更合理
                L_flat_k = _cvm_flatness_value(
                    yk,
                    Z,
                    w_cls_k,
                    n_bins=N_DECOR_BINS,
                    power=2.0,
                )
                flat_penalty += L_flat_k

            flat_penalty *= lam

        total_loss = logloss + flat_penalty
        return float(total_loss)

    return feval
def train_multi_model_un(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    model_name: str,
    decorrelate_feature_names: Optional[Sequence[Union[str, int]]] = None,  # ← 指定需“无关”的变量（用名字，或索引）
):
    """
    多分类（BKG/VVV/VH/...）训练：在标准多类 logloss 的基础上，
    对指定的 'decorrelate_feature_names' 变量与各类别 raw logit 的 Pearson 相关性加入惩罚项，
    并且这些变量不作为训练输入特征（仅用于惩罚项）。

    返回：
      clf, (X_train_used, X_test_used, y_train, y_test, w_train, w_test)
    """
    X = np.asarray(X)
    y = np.asarray(y)
    w = np.asarray(w, dtype=float)

    # 0) 将原始标签映射为 3→5 类等（与你现有实现一致）
    y5, valid_mask = _map_to_3classes(y)  # 保持调用与返回不变
    X, y5, w = X[valid_mask], y5[valid_mask], w[valid_mask]

    # 1) 先划分 train/test（在“剔除列”之前做，以便 Z_train/Z_test 与训练/验证严格同序）
    X_train_all, X_test_all, y_train, y_test, w_train, w_test = train_test_split(
        X, y5, w,
        test_size=0.3,
        stratify=y5,
        random_state=RANDOM_STATE
    )

    # 2) 解析去相关列索引，并构造“训练输入特征”与“仅用于惩罚的 Z”
    decor_idx = _resolve_decor_indices(X_train_all, decorrelate_feature_names)
    if len(decor_idx) > 0:
        all_idx = np.arange(X_train_all.shape[1])
        keep_idx = np.setdiff1d(all_idx, np.array(decor_idx, dtype=int), assume_unique=False)
        if keep_idx.size == 0:
            raise ValueError("去相关列覆盖了所有特征：没有可用于训练的输入。请减少去相关变量数量。")

        # 训练输入（剔除了去相关变量）
        X_train = X_train_all[:, keep_idx]
        X_test  = X_test_all[:, keep_idx]
        # 仅用于惩罚的 Z
        Z_train = X_train_all[:, decor_idx]
        Z_test  = X_test_all[:, decor_idx]
    else:
        # 未指定去相关变量：退化为原始训练
        X_train, X_test = X_train_all, X_test_all
        Z_train = np.zeros((X_train.shape[0], 0), dtype=float)  # shape (n, 0)
        Z_test  = np.zeros((X_test.shape[0], 0), dtype=float)

    # 在真正训练前，打印模型实际看到的输入维度
    print(f"[train_multi_model_un] X_train shape used as model input: {X_train.shape}")
    print(f"[train_multi_model_un] #features used by model: {X_train.shape[1]}")
    print(f"[train_multi_model_un] Z_train shape (decor vars for penalty): {Z_train.shape}")

    
    # 3) 模型超参（保持原有逻辑）
    n_est = 200
    max_depth = 10
    gamma = 0.0
    learning_rate = 0.1
    reg_lambda = 1.0
    reg_alpha = 0.0
    #max_delta_step = 0.0
    min_child_weight = 1.0

    if 'fat2' in model_name:
        n_est = 1000
        max_depth = 14
        gamma = 1
        learning_rate = 0.01
        reg_lambda = 1
        reg_alpha = 1
        #max_delta_step = 20
        min_child_weight = 3
    elif 'fat3' in model_name:
        n_est = 1000
        max_depth = 8
        gamma = 5
        learning_rate = 0.01
        reg_lambda = 10
        reg_alpha = 10
        #max_delta_step = 10
        min_child_weight = 3
    else:
        # (保持上面的默认值即可)
        pass

    # === 通用 XGBoost 参数（不改变你现有的训练超参） ===
    # n_jobs 用一个相对合理的线程数，避免 2048 造成调度反而变慢
    max_threads = os.cpu_count() or 1
    n_threads = max(1, min(16, max_threads))
    print(f"n_threads: {n_threads}")

    
    # 4) 构建分类器
    num_classes = 5  # 与原实现一致
    custom_obj = None

    common_xgb_kwargs = dict(
        num_class=num_classes,
        n_estimators=n_est,
        max_depth=max_depth,
        learning_rate=learning_rate,
        gamma=gamma,
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
        min_child_weight=min_child_weight,
        #max_delta_step=max_delta_step,
        n_jobs=n_threads,
        random_state=RANDOM_STATE,
    )

    # 优先尝试 GPU（需要编译了 GPU 支持的 xgboost），失败则回退到 CPU hist
    gpu_kwargs = dict(
        tree_method="hist",
        #predictor="gpu_predictor",
        device = "cuda"
    )
    cpu_kwargs = dict(
        tree_method="hist",   # CPU 上使用高效直方图算法
    )

    if Z_train.shape[1] > 0 and DECOR_LAMBDA > 0.0:
        # 自定义目标：多类 logloss + decor penalty + 动态学习率
        custom_obj = _make_multiclass_objective_with_decor(
            num_class=num_classes,
            Z_train=Z_train,
            w_train=w_train,
            lam=DECOR_LAMBDA,
            #base_learning_rate=learning_rate,
        )

        # 自定义 eval_metric：总损失（logloss + decor）
        total_loss_metric = _make_multiclass_total_loss_metric(
            num_class=num_classes,
            Z_train=Z_train,
            Z_test=Z_test,
            w_train=w_train,
            w_test=w_test,
            lam=DECOR_LAMBDA,
        )

        # === 优先尝试 GPU 版本，失败则自动回退到 CPU-hist ===
        try:
            clf = XGBClassifier(
                objective=custom_obj,
                eval_metric=total_loss_metric,
                early_stopping_rounds=10,
                **common_xgb_kwargs,
                **gpu_kwargs,
            )
        except xgb.core.XGBoostError:
            clf = XGBClassifier(
                objective=custom_obj,
                eval_metric=total_loss_metric,
                early_stopping_rounds=10,
                **common_xgb_kwargs,
                **cpu_kwargs,
            )

    else:
        # 未去相关时，保持原本的多类 softprob
        try:
            clf = XGBClassifier(
                objective="multi:softprob",
                early_stopping_rounds=10,
                **common_xgb_kwargs,
                **gpu_kwargs,
            )
        except xgb.core.XGBoostError:
            clf = XGBClassifier(
                objective="multi:softprob",
                early_stopping_rounds=10,
                **common_xgb_kwargs,
                **cpu_kwargs,
            )

    # 5) 训练（decor 分支权重在 objective/metric 内部处理）
    if Z_train.shape[1] > 0 and DECOR_LAMBDA > 0.0 and custom_obj is not None:
        clf.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=True
        )
    else:
        clf.fit(
            X_train, y_train,
            sample_weight=w_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            sample_weight_eval_set=[w_train, w_test],
            verbose=True
        )

    # 6) 保存模型
    if Z_train.shape[1] > 0 and DECOR_LAMBDA > 0.0 and custom_obj is not None:
        clf.save_model(model_name if model_name.endswith(".json") else model_name + ".json")
    else:
        with open(model_name, "wb") as fout:
            pickle.dump(clf, fout)

    # 返回的 X_train/X_test 即为“实际用于训练的特征”（已剔除去相关列）
    return clf, (X_train_all, X_test_all, y_train, y_test, w_train, w_test)


def plot_multi_results_un(clf, splits, tree_name, decorrelate_feature_names=None):
    """
    适配 5-class BDT 的可视化：
      (1) ROC：VVV 与其余 4 类分别的 ROC；VH 与其余 4 类分别的 ROC。
      (2) 特征重要性：用真实特征名标注（与训练输入一致），x 轴 log。
      (3) Score 分布：VVV vs rest、VH vs rest（one-vs-rest）。
      (4) Loss 曲线、特征相关矩阵（基于训练输入）。
      (5) 去相关检验（仅此处使用“包含 decor 特征”的 X_full）：
          5.1 【加权】相关系数热力图：5 个 class 的 score vs decor 变量（Train/Test）
          5.2 【加权】scatter：5 个 class 的 score vs decor 变量（Test）
          5.3 【加权】decor 变量分布：对每个 true class，用该 class 自己的 score 做阈值选择，比较 decor 分布（Test）
    """

    X_train_full, X_test_full, y_train, y_test, w_train, w_test = splits

    # ====== 0) 取得“全特征名”与 decor 索引；构造“训练使用的特征名/矩阵” ======
    def _get_full_feature_names(X_like):
        if hasattr(X_like, "columns"):
            return list(X_like.columns)
        if tree_name == "fat2":
            return list(branches2)
        elif tree_name == "fat3":
            return list(branches3)
        ncols = X_like.shape[1]
        return [f"f{i}" for i in range(ncols)]

    full_feature_names = _get_full_feature_names(X_train_full)

    def _resolve_indices(names_or_idx, feature_names_list):
        if not names_or_idx:
            return []
        name_to_idx = {c: i for i, c in enumerate(feature_names_list)}
        out = []
        for key in names_or_idx:
            if isinstance(key, int):
                if 0 <= key < len(feature_names_list):
                    out.append(key)
                else:
                    print(f"[WARN] decor 列号 {key} 越界，已跳过。")
            else:
                if key in name_to_idx:
                    out.append(name_to_idx[key])
                else:
                    print(f"[INFO] decor 变量 '{key}' 不在全特征列表中，跳过。")
        seen, res = set(), []
        for i in out:
            if i not in seen:
                seen.add(i)
                res.append(i)
        return res

    decor_idx_full = _resolve_indices(decorrelate_feature_names, full_feature_names)
    all_idx = np.arange(len(full_feature_names))
    keep_idx = np.setdiff1d(all_idx, np.array(decor_idx_full, dtype=int))  # 训练真正使用的列

    def _slice_cols(X_like, idx):
        if hasattr(X_like, "iloc"):
            return X_like.iloc[:, idx]
        return X_like[:, idx]

    X_train_used = _slice_cols(X_train_full, keep_idx)
    X_test_used  = _slice_cols(X_test_full,  keep_idx)

    feat_names_used = [full_feature_names[i] for i in keep_idx]

    if hasattr(clf, "n_features_in_") and getattr(clf, "n_features_in_") != X_train_used.shape[1]:
        if hasattr(X_train_full, "shape") and X_train_full.shape[1] == getattr(clf, "n_features_in_"):
            X_train_used, X_test_used = X_train_full, X_test_full
            feat_names_used = full_feature_names
            decor_idx_full = _resolve_indices([], full_feature_names)  # decor 检验将被跳过
            print("[INFO] 检测到 splits 已与模型输入对齐，按原 X 使用。")
        else:
            raise ValueError(
                f"特征数与模型不一致：X_used={X_train_used.shape[1]} vs clf.n_features_in_={getattr(clf,'n_features_in_',None)}"
            )

    # ====== 1) 常量与工具 ======
    IDX_VVV, IDX_VH, IDX_TT, IDX_VV, IDX_QCD = 0, 1, 2, 3, 4
    class_names = ["VVV", "VH", "TT", "VV", "QCD"]
    n_classes = len(class_names)

    def _as_array(Xlike):
        return Xlike.to_numpy() if hasattr(Xlike, "to_numpy") else np.asarray(Xlike)

    def _safe_w(w):
        # 相关性/绘图中，权重只作为“统计强度”；如果未来出现负权重，也不至于把分布/相关性搞反
        w = np.asarray(w, float).ravel()
        w = np.abs(w)
        return w

    def _weighted_pearson(x, y, w, eps=1e-12):
        x = np.asarray(x, float).ravel()
        y = np.asarray(y, float).ravel()
        w = _safe_w(w)

        m = np.isfinite(x) & np.isfinite(y) & np.isfinite(w)
        if not np.any(m):
            return 0.0
        x = x[m]
        y = y[m]
        w = w[m]

        sw = w.sum()
        if sw <= eps:
            return 0.0

        wx = (w * x).sum() / (sw + eps)
        wy = (w * y).sum() / (sw + eps)
        x0 = x - wx
        y0 = y - wy
        cov = (w * x0 * y0).sum() / (sw + eps)
        vx = (w * x0 * x0).sum() / (sw + eps)
        vy = (w * y0 * y0).sum() / (sw + eps)
        return float(cov / (np.sqrt(vx * vy) + eps))

    def _weighted_quantile(x, q, w, eps=1e-12):
        """
        加权分位数：q in [0,1] or array-like
        """
        x = np.asarray(x, float).ravel()
        w = _safe_w(w)
        q = np.asarray(q, float)

        m = np.isfinite(x) & np.isfinite(w)
        if not np.any(m):
            return np.full_like(q, np.nan, dtype=float)

        x = x[m]
        w = w[m]
        sw = w.sum()
        if sw <= eps:
            return np.quantile(x, q)

        order = np.argsort(x)
        x_sorted = x[order]
        w_sorted = w[order]
        cw = np.cumsum(w_sorted)
        cw = cw / (cw[-1] + eps)

        return np.interp(q, cw, x_sorted)

    # ====== 2) ROC（用剔除 decor 的 X_*_used）======
    def _roc_one_vs_one(Xs, ys, ws, target_idx, opp_idx):
        probs = clf.predict_proba(Xs)  # shape: (n, 5)
        mask = (ys == target_idx) | (ys == opp_idx)
        if not np.any(mask):
            return None
        y_bin = (ys[mask] == target_idx).astype(int)
        if (y_bin.sum() == 0) or (y_bin.sum() == len(y_bin)):
            return None
        p_t = probs[mask, target_idx]
        p_o = probs[mask, opp_idx]
        eps = 1e-12
        score = p_t / np.clip(p_t + p_o, eps, None)
        auc = roc_auc_score(y_bin, score, sample_weight=ws[mask])
        fpr, tpr, _ = roc_curve(y_bin, score, sample_weight=ws[mask])
        return fpr, tpr, auc

    palette = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple']

    def _plot_all_opponents_for_target(target_idx, target_name):
        opponents = [i for i in range(n_classes) if i != target_idx]
        plt.figure(figsize=(10, 10))
        for color, opp_idx in zip(palette, opponents):
            opp_name = class_names[opp_idx]
            r_test = _roc_one_vs_one(X_test_used, y_test, w_test, target_idx, opp_idx)
            r_train = _roc_one_vs_one(X_train_used, y_train, w_train, target_idx, opp_idx)
            if r_test is not None:
                fpr_tst, tpr_tst, auc_tst = r_test
                plt.plot(tpr_tst, fpr_tst, color=color, linestyle='-',
                         label=f"{opp_name} (Test AUC={auc_tst:.3f})")
                print(f"{tree_name} Test AUC ({target_name} vs {opp_name}) = {auc_tst:.4f}")
            else:
                print(f"{tree_name} Test AUC ({target_name} vs {opp_name}) = N/A")
            if r_train is not None:
                fpr_tr, tpr_tr, auc_tr = r_train
                plt.plot(tpr_tr, fpr_tr, color=color, linestyle='--',
                         label=f"{opp_name} (Train AUC={auc_tr:.3f})")
                print(f"{tree_name} Train AUC ({target_name} vs {opp_name}) = {auc_tr:.4f}")
            else:
                print(f"{tree_name} Train AUC ({target_name} vs {opp_name}) = N/A")
        plt.xlabel(rf"$\epsilon_{{\rm {target_name}}}$", fontsize=20)
        plt.ylabel(r"$\epsilon_{\rm bkg}$", fontsize=20)
        plt.yscale("log")
        plt.ylim(1e-6, 1)
        plt.xlim(0, 1)
        plt.legend(loc="lower right", fontsize=22)
        plt.tight_layout()
        plt.savefig(f"{tree_name}_ROC_{target_name}_vs_all.png")

    _plot_all_opponents_for_target(IDX_VVV, "VVV")
    # _plot_all_opponents_for_target(IDX_VH, "VH")

    # ====== 3) 特征重要性（用真实特征名，且与训练输入一致）======
    fig, ax = plt.subplots(figsize=(10, 20))
    plot_importance(clf, ax=ax, max_num_features=200)
    ax.set_title(f"{tree_name} Feature Importance", fontsize=16)
    ax.set_xscale("log")
    widths = [patch.get_width() for patch in ax.patches]
    if len([w for w in widths if w > 0]) > 0:
        min_nonzero = min(w for w in widths if w > 0)
        max_width = max(widths) if widths else 1.0
        ax.set_xlim(min_nonzero / 2, max_width * 2)

    try:
        raw_labels = [t.get_text() for t in ax.get_yticklabels()]
        mapped = []
        for s in raw_labels:
            if isinstance(s, str) and s.startswith('f'):
                num = s[1:]
                if num.isdigit():
                    i = int(num)
                    mapped.append(feat_names_used[i] if i < len(feat_names_used) else s)
                else:
                    mapped.append(s)
            else:
                mapped.append(s)
        if mapped:
            ax.set_yticklabels(mapped)
    except Exception:
        pass
    plt.tight_layout()
    plt.savefig(f"{tree_name}_Feature_Importance.png")

    # ====== 4) Score 分布（基于剔除 decor 的预测）======
    probs_train = clf.predict_proba(X_train_used)
    probs_test  = clf.predict_proba(X_test_used)

    def _plot_score_dist_for_target(target_idx, target_name, bkg_name):
        p_train = probs_train[:, target_idx]
        p_test  = probs_test[:, target_idx]

        train_pos   = p_train[y_train == target_idx]
        train_w_pos = w_train[y_train == target_idx]
        train_bkg   = p_train[y_train != target_idx]
        train_w_bkg = w_train[y_train != target_idx]

        test_pos   = p_test[y_test == target_idx]
        test_w_pos = w_test[y_test == target_idx]
        test_bkg   = p_test[y_test != target_idx]
        test_w_bkg = w_test[y_test != target_idx]

        plt.figure()
        plt.xlim(0, 1)
        bin_edges = np.linspace(0, 1, 31)
        plt.hist(train_bkg, bins=bin_edges, weights=train_w_bkg,
                 density=True, histtype="bar", alpha=0.5, label=f"Train {bkg_name}")
        plt.hist(train_pos, bins=bin_edges, weights=train_w_pos,
                 density=True, histtype="bar", alpha=0.5, label=f"Train {target_name}")
        plt.hist(test_bkg, bins=bin_edges, weights=test_w_bkg,
                 density=True, histtype="step", linewidth=2, alpha=0.7,
                 label=f"Test {bkg_name}", color="lime")
        plt.hist(test_pos, bins=bin_edges, weights=test_w_pos,
                 density=True, histtype="step", linewidth=2, alpha=0.7,
                 label=f"Test {target_name}", color="red")
        plt.xlabel("BDT Score")
        plt.yscale("log")
        plt.ylim(1e-2,)
        plt.ylabel("Density")
        #plt.title(f"{tree_name} Score Distribution ({target_name} vs {bkg_name})", fontsize=16)
        plt.legend()
        plt.savefig(f"{tree_name}_Score_Dist_{target_name}_vs_all.png")

    _plot_score_dist_for_target(IDX_VVV, "VVV", "VH+VV+TT+QCD")
    _plot_score_dist_for_target(IDX_VH,  "VH",  "VVV+VV+TT+QCD")

    # ====== 5) Loss 曲线（总 loss）======
    try:
        evals = clf.evals_result() if hasattr(clf, 'evals_result') else clf.evals_result_
    except Exception:
        evals = None

    if evals is not None and 'validation_0' in evals and 'validation_1' in evals:
        keys0 = list(evals['validation_0'].keys())
        loss_key = None
        for k in keys0:
            if 'total_loss' in k:
                loss_key = k
                break
        if loss_key is None:
            for k in keys0:
                if 'feval' in k:
                    loss_key = k
                    break
        if loss_key is None:
            for k in keys0:
                if 'logloss' in k:
                    loss_key = k
                    break

        if loss_key is not None:
            train_loss = evals['validation_0'][loss_key]
            test_loss  = evals['validation_1'][loss_key]
            epochs = range(1, len(train_loss) + 1)

            plt.figure(figsize=(8, 5))
            plt.plot(epochs, train_loss, label='Train')
            plt.plot(epochs, test_loss,  label='Test')
            plt.xlabel('Epoch')
            plt.ylabel('total_loss')
            #plt.title(f'{tree_name} Total Loss vs. Epoch')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.savefig(f"{tree_name}_Total_Loss_Curve.png")

    # ====== 6) 特征相关矩阵（基于训练输入的特征）======
    X_train_used_df = pd.DataFrame(_as_array(X_train_used), columns=feat_names_used)
    corr = X_train_used_df.corr(numeric_only=True)
    corr = corr.dropna(axis=0, how='all').dropna(axis=1, how='all')
    plt.figure(figsize=(20, 20))
    plt.imshow(corr.values, aspect='equal', interpolation='none', vmin=-1, vmax=1, cmap='bwr')
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=10)
    plt.yticks(range(len(corr.index)),    corr.index, fontsize=10)
    #plt.title(f"{tree_name} Feature Correlation Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{tree_name}_Feature_Correlation_Matrix.png")

    # ====== 7) 【去相关检验】仅此处使用“包含 decor 的 X_full” ======
    if len(decor_idx_full) == 0:
        print("[INFO] 未找到可用于去相关检验的变量（可能未指定或训练时已全部剔除）。")
        return

    decor_var_names = [full_feature_names[i] for i in decor_idx_full]

    # 先得到概率（概率基于“训练输入”，decor 值来自“全输入”）
    probs_train = clf.predict_proba(X_train_used)
    probs_test  = clf.predict_proba(X_test_used)

    # ---------- 7.1 【加权】相关系数矩阵：行=class score，列=decor 变量；分 Train/Test ----------
    def _build_corr_matrix(probs, Xmat_full, wvec):
        Xarr = _as_array(Xmat_full)
        wv = _safe_w(wvec)
        R = np.zeros((n_classes, len(decor_idx_full)), dtype=float)
        for r, ci in enumerate(range(n_classes)):
            s = probs[:, ci]
            for c, j in enumerate(decor_idx_full):
                x = Xarr[:, j]
                R[r, c] = _weighted_pearson(x, s, wv)
        return R

    R_train = _build_corr_matrix(probs_train, X_train_full, w_train)
    R_test  = _build_corr_matrix(probs_test,  X_test_full,  w_test)

    def _plot_corr_heatmap(R, title):
        plt.figure(figsize=(1.2 * len(decor_idx_full) + 6, 1.0 * n_classes + 4))
        plt.imshow(R, aspect='auto', interpolation='none', vmin=-1, vmax=1, cmap='bwr')
        cbar = plt.colorbar(fraction=0.046, pad=0.04, label="weighted Pearson r")
        cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
        plt.xticks(range(len(decor_var_names)), decor_var_names, rotation=45, ha='right', fontsize=11)
        plt.yticks(range(n_classes), class_names, fontsize=12)
        for i in range(n_classes):
            for j in range(len(decor_var_names)):
                val = R[i, j]
                plt.text(j, i, f"{val:+.2f}", ha='center', va='center',
                         color='white' if abs(val) > 0.5 else 'black', fontsize=10)
        #plt.title(title, fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{tree_name}_Decor_Check_Heatmap_{title.replace(' ','_').replace(':','')}.png")

    _plot_corr_heatmap(R_train, f"{tree_name} Decor-Check Heatmap (Train): class score vs decor")
    _plot_corr_heatmap(R_test,  f"{tree_name} Decor-Check Heatmap (Test): class score vs decor")

    # ---------- 7.2 【加权】decor 变量分布：每个 true class，用该 class 自己的 score 做“效率阈值”选择（Test） ----------
    try:
        X_test_arr = _as_array(X_test_full)
        y_all = np.asarray(y_test, int)
        w_all = _safe_w(w_test)

        # 这里改为按“剩余效率”定义的阈值：大于 threshold 后剩下约 [100, 10, 1, 0.5, 0.1] %
        eff_targets = np.array([1.0, 0.5, 0.1, 0.01], dtype=float)

        # 颜色：沿用当前 CMS/Matplotlib 的默认循环
        try:
            color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        except Exception:
            color_cycle = []
        if len(color_cycle) < len(eff_targets):
            color_cycle = [f"C{i}" for i in range(len(eff_targets))]
        colors = color_cycle[:len(eff_targets)]

        for j, var_name in zip(decor_idx_full, decor_var_names):
            x_all = X_test_arr[:, j].astype(float)

            for class_idx, class_name in enumerate(class_names):
                mask_cls = (y_all == class_idx)
                if not np.any(mask_cls):
                    continue

                # 该 class 自己的 score（只在该 class 内决定阈值，使得“剩余效率”满足 eff_targets）
                score_k = probs_test[:, class_idx].astype(float)
                score_cls = score_k[mask_cls]
                w_cls = w_all[mask_cls]

                if (score_cls.size == 0) or (w_cls.sum() <= 0) or (not np.any(np.isfinite(score_cls))):
                    continue

                # 为这个 true class 计算对应的 score 阈值
                thr_list = []
                for eff in eff_targets:
                    # eff 是“score > thr”后期望保留的加权效率
                    if eff >= 1.0 - 1e-12:
                        # 100% 效率：用一个远小于 score 的值，等价于“无 cut”
                        thr = -1e9
                    else:
                        q = 1.0 - eff  # “<= thr”的分位数
                        thr = float(_weighted_quantile(score_cls, [q], w_cls)[0])
                    thr_list.append(thr)

                # 取该 true class 内 decor 的加权分位数做 x 轴范围，避免被长尾撑爆
                x_cls = x_all[mask_cls]
                if not np.any(np.isfinite(x_cls)):
                    continue

                qlo, qhi = _weighted_quantile(x_cls, [0.01, 0.99], w_cls)
                if not np.isfinite(qlo) or not np.isfinite(qhi) or qlo == qhi:
                    # fallback
                    qlo = np.nanmin(x_cls)
                    qhi = np.nanmax(x_cls)
                    if not np.isfinite(qlo) or not np.isfinite(qhi) or qlo == qhi:
                        continue

                # 给一点边界余量
                span = float(qhi - qlo)
                xmin = float(qlo - 0.02 * span)
                xmax = float(qhi + 0.02 * span)
                # 统一 bin
                n_bins = 100
                bin_edges = np.linspace(xmin, xmax, n_bins + 1)

                plt.figure(figsize=(8, 6))
                for thr, eff, c in zip(thr_list, eff_targets, colors):
                    sel = mask_cls & (score_k > thr) & np.isfinite(x_all) & np.isfinite(w_all)
                    if not np.any(sel):
                        continue
                    plt.hist(
                        x_all[sel],
                        bins=bin_edges,
                        weights=w_all[sel],
                        density=True,
                        histtype="step",
                        linewidth=2.0,
                        label=f"Efficiency {eff*100:g} %)",
                        color=c,
                    )

                plt.xlabel(var_name)
                plt.ylabel("A.U.")
                ##plt.title(f"{class_name")
                plt.legend()
                plt.tight_layout()
                plt.savefig(f"{tree_name}_Decor_Check_{var_name}_by_{class_name}_Test.png")

    except Exception as e:
        print(f"[WARN] decor 变量分布绘图时发生异常: {e}")

    # ---------- 7.3 【加权】散点图：对每个 class score vs 每个 decor 变量（Test） ----------
    def _scatter_var_vs_score_all_classes(Xmat_full, probs, wvec, var_idx, var_name):
        Xarr = _as_array(Xmat_full)
        x0 = Xarr[:, var_idx].astype(float)
        w0 = _safe_w(wvec)

        # 统一过滤非有限
        base_m = np.isfinite(x0) & np.isfinite(w0)
        if not np.any(base_m):
            return

        # 预先做一次下采样索引（对所有 class 共用同一批点，便于对比）
        x = x0[base_m]
        w = w0[base_m]
        idx_base = np.nonzero(base_m)[0]

        n = len(x)
        max_n = 50000
        if n > max_n:
            p = w / (w.sum() + 1e-12)
            choose = np.random.RandomState(7).choice(np.arange(n), size=max_n, replace=False, p=p)
            idx_sel = idx_base[choose]
        else:
            idx_sel = idx_base

        x_s = x0[idx_sel]
        w_s = w0[idx_sel]

        # 用加权分位数给 x 轴范围，避免长尾
        qlo, qhi = _weighted_quantile(x_s, [0.01, 0.99], w_s)
        if np.isfinite(qlo) and np.isfinite(qhi) and qlo < qhi:
            span = float(qhi - qlo)
            xmin = float(qlo - 0.05 * span)
            xmax = float(qhi + 0.05 * span)
        else:
            xmin = np.nanmin(x_s)
            xmax = np.nanmax(x_s)
        if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin == xmax:
            return

        # 对每个 class 单独画一张
        for class_idx, class_name in enumerate(class_names):
            y0 = probs[:, class_idx].astype(float)
            y_s = y0[idx_sel]

            m = np.isfinite(x_s) & np.isfinite(y_s) & np.isfinite(w_s)
            if not np.any(m):
                continue
            xs = x_s[m]
            ys = y_s[m]
            ws = w_s[m]

            plt.figure(figsize=(7, 5))
            plt.scatter(
                xs, ys,
                s=np.clip(10.0 * (ws / (ws.max() + 1e-12)), 2, 30),
                alpha=0.25,
                edgecolors='none'
            )

            # 黑色均值线：按“加权分位数”分箱，每箱内画 y 的加权平均
            try:
                qs = np.linspace(0, 1, 21)
                edges = _weighted_quantile(xs, qs, ws)
                edges[0] -= 1e-12
                edges[-1] += 1e-12

                xc, yc = [], []
                for i in range(len(edges) - 1):
                    lo, hi = edges[i], edges[i + 1]
                    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
                        continue
                    mm = (xs >= lo) & (xs < hi) if i < len(edges) - 2 else (xs >= lo) & (xs <= hi)
                    if not np.any(mm):
                        continue
                    wsum = ws[mm].sum()
                    if wsum <= 1e-12:
                        continue
                    xc.append(0.5 * (lo + hi))
                    yc.append((ws[mm] * ys[mm]).sum() / (wsum + 1e-12))
                if xc:
                    plt.plot(xc, yc, linewidth=2.0, color="black")
            except Exception:
                pass

            plt.xlim(xmin, xmax)
            plt.xlabel(var_name)
            plt.ylabel(f"{class_name} score")
            #plt.title(f"{tree_name} Decor-Check Scatter (Test): {var_name} vs {class_name} score")
            plt.tight_layout()
            plt.savefig(f"{tree_name}_Decor_Check_Scatter_{var_name}_vs_{class_name}_Test.png")

    for j, vname in zip(decor_idx_full, decor_var_names):
        _scatter_var_vs_score_all_classes(X_test_full, probs_test, w_test, j, vname)


X2, y2, w2, X2_data, y2_data, w2_data = prepare_multi_data('fat2', branches2, ENTRIES_PER_SAMPLE * 2)
thresholds = {
    'msd8_1': (40,  150),
    'msd8_2': (40,  150),
    'pt8_1' : (180, None),
    'pt8_2' : (180, None),
    'eta8_1' : (-2.4, 2.4),
    'eta8_2' : (-2.4, 2.4),
    #'sphereM' : (0, 900)
}
    
X2, y2, w2 = filter_X(X2, y2, w2, branches2, thresholds, apply_to_sentinel=True)
X2_data, y2_data, w2_data = filter_X(X2_data, y2_data, w2_data, branches2, thresholds, apply_to_sentinel=True)
X2_std = standardize_X(X2)

#check_weights(w2, "w2")
FEATURE_NAMES = list(X2_std.columns)
ix1 = X2_std.columns.get_loc("msd8_1")
clf2, splits2 = train_multi_model_un(X2_std, y2, w2, 'bdt_fat2_nocorr_v4', decorrelate_feature_names=[ix1])
#print(X2_std)
plot_multi_results_un(clf2, splits2, 'fat2', decorrelate_feature_names=["msd8_1"])
