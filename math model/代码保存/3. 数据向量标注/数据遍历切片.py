import os
import re
import csv
import argparse
from pathlib import Path
import numpy as np
from scipy.io import loadmat, savemat
from scipy.signal import butter, filtfilt, hilbert, detrend

# -----------------------------
# 你提供的 angle_slice 函数（略有注释）
# -----------------------------
def angle_slice(
    x, fs, rpm=None, fr_estimator=None,
    K=512, R=32, overlap=0.5, smooth=None
):
    """
    x: 1D numpy array
    fs: sampling rate (Hz)
    rpm: None / scalar / 1D array (与 x 同步或较稀疏都行)
    fr_estimator: 函数 f(x, fs) -> fr(Hz)（若 rpm=None 且希望估计）
    K: 每转采样点
    R: 每片包含的转数
    overlap: 片间重叠比例（0~<1）
    smooth: None 或 ('median', win) / ('savgol', win, poly)
    return: segments, shape = [num_segments, R, K]
    """
    x = np.asarray(x).astype(float)
    N = len(x)
    t = np.arange(N) / fs

    # --- 1) 转频序列 fr(t) ---
    if rpm is not None:
        if np.ndim(rpm) == 0:
            fr = np.full(N, rpm / 60.0)
        else:
            rpm = np.asarray(rpm).astype(float)
            t_rpm = np.linspace(0, t[-1], len(rpm))
            fr = np.interp(t, t_rpm, rpm) / 60.0
    else:
        if fr_estimator is None:
            raise ValueError("rpm 未提供时需给出 fr_estimator(x, fs)")
        fr0 = float(fr_estimator(x, fs))  # 常数转频估计
        fr = np.full(N, fr0)

    # 可选平滑
    if smooth is not None:
        kind = smooth[0]
        if kind == 'median':
            from scipy.signal import medfilt
            win = smooth[1] if len(smooth) > 1 else 101
            if win % 2 == 0: win += 1
            fr = medfilt(fr, kernel_size=win)
        elif kind == 'savgol':
            from scipy.signal import savgol_filter
            win = smooth[1] if len(smooth) > 1 else 101
            poly = smooth[2] if len(smooth) > 2 else 3
            if win % 2 == 0: win += 1
            fr = savgol_filter(fr, window_length=win, polyorder=poly, mode="interp")

    # --- 2) 角度重采样 ---
    rot = np.cumsum(fr) / fs                 # 累计转数
    total_turns = rot[-1]
    num_turns = int(np.floor(total_turns))
    if num_turns < R:
        return np.empty((0, R, K))

    r_grid = np.linspace(0.0, num_turns, num_turns * K, endpoint=False)
    x_angle = np.interp(r_grid, rot, x).reshape(num_turns, K)

    # --- 3) 固定转数切片 ---
    hop = max(1, int(round(R * (1 - overlap))))
    starts = np.arange(0, num_turns - R + 1, hop)

    segments = []
    for s in starts:
        seg = x_angle[s:s+R].copy()          # (R, K)
        mu = seg.mean()
        std = seg.std() + 1e-8               # 片内 z-score
        seg = (seg - mu) / std
        segments.append(seg)
    if len(segments) == 0:
        return np.empty((0, R, K))
    return np.stack(segments, axis=0)        # (num_segments, R, K)

# -----------------------------
# 转频估计（包络谱法）
# -----------------------------
def estimate_fr_envelope(x, fs, lo=2.0, hi=100.0):
    x = np.asarray(x).astype(float)
    # 包络
    analytic = hilbert(x)
    env = np.abs(analytic)
    # FFT
    N = len(env)
    freqs = np.fft.rfftfreq(N, 1/fs)
    spectrum = np.abs(np.fft.rfft(env))
    mask = (freqs >= lo) & (freqs <= hi)
    if not np.any(mask):
        return 10.0  # fallback
    freqs_low, spec_low = freqs[mask], spectrum[mask]
    fr_est = float(freqs_low[np.argmax(spec_low)])
    return fr_est

# -----------------------------
# 信号预处理（可按需开启/关闭）
# -----------------------------
def bandpass(x, fs, lo=500.0, hi=10000.0, order=4):
    # lo 或 hi 若为 None 则退化为高通/低通
    nyq = 0.5 * fs
    if hi is None:           # 高通
        b, a = butter(order, lo/nyq, btype='high')
    elif lo is None:         # 低通
        b, a = butter(order, hi/nyq, btype='low')
    else:                    # 带通
        lo_ = max(1e-6, lo/nyq)
        hi_ = min(0.999, hi/nyq)
        if hi_ <= lo_:
            hi_ = min(0.999, lo_ + 1e-4)
        b, a = butter(order, [lo_, hi_], btype='band')
    return filtfilt(b, a, x)

# -----------------------------
# 工具：推断采样率、找到信号变量、RPM 变量
# -----------------------------
def infer_fs_from_path(path: Path) -> int:
    p = str(path).lower()
    if "12khz" in p or "12_khz" in p or "12 khz" in p:
        return 12000
    if "48khz" in p or "48_khz" in p or "48 khz" in p:
        return 48000
    # 也可以在此扩展 32kHz 等
    raise ValueError(f"无法从路径推断采样率，请在文件名/路径中包含 '12kHz' 或 '48kHz'：{path}")

def load_mat_vars(path: Path) -> dict:
    data = loadmat(str(path), squeeze_me=True, struct_as_record=False)
    # 去掉元变量
    return {k: v for k, v in data.items() if not k.startswith("__")}

def find_signal_vars(d: dict) -> list:
    """
    选择“像时序”的变量：
    - 名称包含 *_DE_time / *_FE_time / *_BA_time，或
    - 数值型 1D 向量，长度 > 1000
    """
    keys = list(d.keys())
    pri = [k for k in keys if re.search(r'(de|fe|ba)_time$', k.lower())]
    out = []
    if pri:
        out.extend(pri)
    for k, v in d.items():
        if k in out: continue
        if isinstance(v, np.ndarray) and v.ndim == 1 and v.size > 1000 and np.issubdtype(v.dtype, np.number):
            out.append(k)
    return out

def find_rpm_var(d: dict) -> np.ndarray | None:
    cand = [k for k in d.keys() if re.search(r'rpm', k, flags=re.IGNORECASE)]
    for k in cand:
        v = d[k]
        if np.isscalar(v) and np.isfinite(v):
            return np.array([float(v)])
        if isinstance(v, np.ndarray) and v.size >= 1:
            vv = np.asarray(v).astype(float).ravel()
            if np.all(np.isfinite(vv)):
                return vv
    return None

# -----------------------------
# 主流程：遍历 -> 读取 -> 处理 -> 保存
# -----------------------------
def process_one_mat(src_path: Path, out_root: Path,
                    K, R, overlap, bp_lo, bp_hi):
    # 读取
    d = load_mat_vars(src_path)

    # 推断 fs
    fs = infer_fs_from_path(src_path)

    # 预处理 & RPM
    rpm_arr = find_rpm_var(d)  # 可为空
    fr_used = None
    if rpm_arr is not None and rpm_arr.size == 1:
        fr_used = float(rpm_arr[0]) / 60.0
    # 输出目录（镜像原路径）
    rel = src_path.relative_to(root)
    dst_dir = out_root / rel.parent
    dst_dir.mkdir(parents=True, exist_ok=True)

    # 找信号变量
    sig_vars = find_signal_vars(d)
    if not sig_vars:
        print(f"[WARN] 未找到时序变量：{src_path}")
        return 0

    total_segs = 0

    for vname in sig_vars:
        x = np.asarray(d[vname]).astype(float).ravel()
        # 可选预处理：去直流 + 带通（按需调整/关闭）
        x = detrend(x, type='constant')
        if (bp_lo is not None) or (bp_hi is not None):
            x = bandpass(x, fs, lo=bp_lo, hi=bp_hi, order=4)

        # angle_slice
        if rpm_arr is not None:
            segs = angle_slice(x, fs, rpm=rpm_arr, fr_estimator=None,
                               K=K, R=R, overlap=overlap, smooth=None)
            if fr_used is None:
                # 若 rpm 是序列，这里给一个代表值（中位数）
                fr_used = float(np.median(rpm_arr) / 60.0)
        else:
            segs = angle_slice(x, fs, rpm=None,
                               fr_estimator=lambda sig, fs_: estimate_fr_envelope(sig, fs_),
                               K=K, R=R, overlap=overlap, smooth=None)
            fr_used = float(estimate_fr_envelope(x, fs))

        nseg = segs.shape[0]
        if nseg == 0:
            print(f"[INFO] 转数不足 R={R}，跳过：{src_path} 变量 {vname}")
            continue

        # 保存每个切片：保持“像原文件”的格式（同名变量），但把 R×K 展平成 1D
        base = src_path.stem  # 原文件名不含扩展
        for i in range(nseg):
            seg = segs[i].reshape(-1)  # (R*K,)
            out_name = f"{base}_{vname}_seg{i:03d}.mat"
            out_path = dst_dir / out_name
            md = {
                vname: seg,
                "RPM": np.array([fr_used * 60.0], dtype=float),
                "fs_src": np.array([fs], dtype=float),
                "K": np.array([K], dtype=int),
                "R": np.array([R], dtype=int),
                "overlap": np.array([overlap], dtype=float),
                "fr_used": np.array([fr_used], dtype=float),
            }
            savemat(str(out_path), md, do_compression=True)
        total_segs += nseg

    return total_segs

def walk_and_process(root: Path, out_root: Path,
                     K, R, overlap, bp_lo, bp_hi, index_csv: Path):
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    cnt_file = 0
    cnt_seg_all = 0

    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if not fn.lower().endswith(".mat"):
                continue
            src_path = Path(dirpath) / fn
            try:
                nseg = process_one_mat(src_path, out_root, K, R, overlap, bp_lo, bp_hi)
                cnt_file += 1
                cnt_seg_all += nseg
                rows.append([str(src_path), nseg])
                print(f"✔ {src_path} -> {nseg} 切片")
            except Exception as e:
                print(f"✖ 处理失败: {src_path}\n   {e}")

    # 写索引
    with open(index_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["src_file", "num_segments"])
        w.writerows(rows)

    print(f"\n✅ 完成：处理 {cnt_file} 个文件，共生成 {cnt_seg_all} 个切片")
    print(f"索引已写出：{index_csv}")

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="遍历 .mat，角度重采样切片并以原格式保存")
    ap.add_argument("root", type=str, help="源数据根目录")
    ap.add_argument("--out_root", type=str, required=True, help="切片输出根目录")
    ap.add_argument("--K", type=int, default=512, help="每转采样点")
    ap.add_argument("--R", type=int, default=32, help="每片包含转数")
    ap.add_argument("--overlap", type=float, default=0.5, help="片间重叠 [0,1)")
    ap.add_argument("--bp_lo", type=float, default=500.0, help="带通下限 Hz（None 关闭）")
    ap.add_argument("--bp_hi", type=float, default=10000.0, help="带通上限 Hz（None 关闭）")
    args = ap.parse_args()

    root = Path(args.root)
    out_root = Path(args.out_root)

    # 将字符串 "None" 视为关闭带通端
    bp_lo = None if (isinstance(args.bp_lo, str) and args.bp_lo.lower()=="none") else args.bp_lo
    bp_hi = None if (isinstance(args.bp_hi, str) and args.bp_hi.lower()=="none") else args.bp_hi

    walk_and_process(root, out_root,
                     K=args.K, R=args.R, overlap=args.overlap,
                     bp_lo=bp_lo, bp_hi=bp_hi,
                     index_csv=out_root / "slices_index.csv")
