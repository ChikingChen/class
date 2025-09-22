import os, re, csv, argparse
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
from scipy.io import loadmat
from scipy.signal import hilbert, butter, filtfilt, detrend

# -------- 角度重采样切片 ----------
def angle_slice(x, fs, rpm=None, fr_estimator=None, K=512, R=32, overlap=0.5, smooth=None):
    x = np.asarray(x).astype(float)
    N = len(x); t = np.arange(N)/fs
    if rpm is not None:
        if np.ndim(rpm)==0: fr = np.full(N, rpm/60.0)
        else:
            rpm = np.asarray(rpm).astype(float)
            t_rpm = np.linspace(0, t[-1], len(rpm))
            fr = np.interp(t, t_rpm, rpm)/60.0
    else:
        if fr_estimator is None: raise ValueError("rpm 未提供时需给出 fr_estimator(x, fs)")
        fr = np.full(N, float(fr_estimator(x, fs)))
    rot = np.cumsum(fr)/fs
    num_turns = int(np.floor(rot[-1]))
    if num_turns < R: return np.empty((0, R, K))
    r_grid = np.linspace(0.0, num_turns, num_turns*K, endpoint=False)
    x_angle = np.interp(r_grid, rot, x).reshape(num_turns, K)
    hop = max(1, int(round(R*(1-overlap))))
    starts = np.arange(0, num_turns-R+1, hop)
    segs=[]
    for s in starts:
        seg = x_angle[s:s+R].copy()
        seg = (seg-seg.mean())/(seg.std()+1e-8)
        segs.append(seg)
    return np.stack(segs, axis=0) if segs else np.empty((0, R, K))

# -------- 转频估计（包络谱） ----------
def estimate_fr_envelope(x, fs, lo=2.0, hi=100.0):
    env = np.abs(hilbert(np.asarray(x).astype(float)))
    N = len(env); freqs = np.fft.rfftfreq(N, 1/fs); spec = np.abs(np.fft.rfft(env))
    mask = (freqs>=lo) & (freqs<=hi)
    return float(freqs[mask][np.argmax(spec[mask])]) if np.any(mask) else 10.0

# -------- 向量/图像生成 ----------
def _interp_to_grid(x_src, y_src, x_dst):
    return np.interp(x_dst, x_src, y_src, left=y_src[0], right=y_src[-1])

def order_vector_from_segment(seg, Omax=50, D=256, agg="mean", log_amp=True, eps=1e-8):
    R, K = seg.shape; orders = np.arange(K//2+1, dtype=float)
    mags = np.empty((R, orders.size), float)
    for r in range(R): mags[r] = np.abs(np.fft.rfft(seg[r]))
    S = np.median(mags,0) if agg=="median" else (mags.sum(0) if agg=="sum" else mags.mean(0))
    if log_amp: S = np.log1p(S)
    mask = orders <= min(Omax, orders[-1]); x_dst = np.linspace(0.0, float(Omax), D)
    vec = _interp_to_grid(orders[mask], S[mask], x_dst)
    vec = (vec-vec.mean())/(vec.std()+eps)
    return vec.astype(np.float32)

def ordergram_from_segment(seg, Omax=50, H=128, W=128, log_amp=True, eps=1e-8):
    R, K = seg.shape; orders = np.arange(K//2+1, dtype=float)
    spec = np.empty((orders.size, R), float)
    for r in range(R): spec[:,r] = np.abs(np.fft.rfft(seg[r]))
    if log_amp: spec = np.log1p(spec)
    mask = orders <= min(Omax, orders[-1]); spec = spec[mask,:]; orders_clip = orders[mask]
    O_dst = np.linspace(0.0, float(Omax), H)
    spec_H = np.stack([_interp_to_grid(orders_clip, spec[:,j], O_dst) for j in range(spec.shape[1])], 1)
    t_src = np.linspace(0.0,1.0,spec_H.shape[1]); t_dst = np.linspace(0.0,1.0,W)
    out = np.stack([_interp_to_grid(t_src, spec_H[i,:], t_dst) for i in range(H)], 0)
    out = (out-out.mean())/(out.std()+eps)
    return out.astype(np.float32)

# -------- 读取 .mat / 采样率 / RPM ----------
def load_mat_vars(path: Path) -> dict:
    d = loadmat(str(path), squeeze_me=True, struct_as_record=False)
    return {k:v for k,v in d.items() if not k.startswith("__")}

def find_signal_vars(d: dict) -> List[str]:
    pri = [k for k in d if re.search(r'(DE|FE|BA)_time$', k, re.IGNORECASE)]
    if pri: return pri
    out=[]
    for k,v in d.items():
        if isinstance(v,np.ndarray) and v.ndim==1 and v.size>1000 and np.issubdtype(v.dtype, np.number):
            out.append(k)
    return out

def try_get_rpm(d: dict, fname: str) -> Optional[np.ndarray]:
    for k,v in d.items():
        if re.search(r'rpm', k, re.IGNORECASE):
            if np.isscalar(v) and np.isfinite(v): return np.array([float(v)])
            if isinstance(v,np.ndarray) and v.size>=1:
                vv=np.asarray(v).astype(float).ravel()
                if np.all(np.isfinite(vv)): return vv
    m = re.search(r'\((\d+)\s*rpm\)', fname.lower())
    return np.array([float(m.group(1))]) if m else None

def infer_fs_from_path(path: Path) -> int:
    p=str(path).lower()
    if "12khz" in p: return 12000
    if "48khz" in p: return 48000
    raise ValueError(f"无法从路径推断采样率（需含 12kHz/48kHz）：{path}")

def bandpass(x, fs, lo=500.0, hi=10000.0, order=4):
    from scipy.signal import butter, filtfilt
    if lo is None and hi is None: return x
    nyq=0.5*fs
    if hi is None:
        b,a=butter(order, lo/nyq, btype='high')
    elif lo is None:
        b,a=butter(order, hi/nyq, btype='low')
    else:
        lo_=max(1e-6,lo/nyq); hi_=min(0.999,hi/nyq)
        if hi_<=lo_: hi_=min(0.999, lo_+1e-4)
        b,a=butter(order,[lo_,hi_],btype='band')
    return filtfilt(b,a,x)

# -------- 单文件处理（仅 .npy 输出） ----------
def process_one_file(src: Path, out_root: Path, mode: str,
                     K:int,R:int,overlap:float,Omax:int,D:int,H:int,W:int,
                     bp_lo:Optional[float], bp_hi:Optional[float]) -> List[Tuple[str,int,float,str]]:
    d = load_mat_vars(src)
    fs = infer_fs_from_path(src)
    rpm = try_get_rpm(d, src.name)  # None / 标量 / 序列
    sig_vars = find_signal_vars(d)
    if not sig_vars:
        print(f"[WARN] 无时序变量: {src}")
        return []

    rel = src.relative_to(root)
    out_dir = out_root / rel.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rows=[]
    for vname in sig_vars:
        x = np.asarray(d[vname]).astype(float).ravel()
        x = detrend(x, type="constant")
        if bp_lo is not None or bp_hi is not None:
            x = bandpass(x, fs, bp_lo, bp_hi)

        if rpm is None:
            segs = angle_slice(x, fs, rpm=None,
                               fr_estimator=lambda sig,fs_: estimate_fr_envelope(sig,fs_),
                               K=K,R=R,overlap=overlap)
            fr_used = float(estimate_fr_envelope(x, fs))
        else:
            segs = angle_slice(x, fs, rpm=rpm, fr_estimator=None,
                               K=K,R=R,overlap=overlap)
            fr_used = float((np.median(rpm) if rpm.size>1 else rpm.item())/60.0)

        if segs.shape[0]==0:
            print(f"[INFO] 转数不足 R={R}: {src} / {vname}")
            continue

        base = f"{src.stem}_{vname}"
        if mode=="vector":
            vecs = np.stack([order_vector_from_segment(seg,Omax=Omax,D=D) for seg in segs],0)
            out_path = out_dir / f"{base}_vecs.npy"
            np.save(out_path, vecs)
            rows.append([str(src), vname, segs.shape[0], fr_used*60.0, "vector", str(out_path)])
        else:
            imgs = np.stack([ordergram_from_segment(seg,Omax=Omax,H=H,W=W) for seg in segs],0)
            out_path = out_dir / f"{base}_imgs.npy"
            np.save(out_path, imgs)
            rows.append([str(src), vname, segs.shape[0], fr_used*60.0, "image", str(out_path)])

    return rows

# -------- 遍历整棵树 ----------
def walk_and_process(root: Path, out_root: Path, mode: str,
                     K:int,R:int,overlap:float,Omax:int,D:int,H:int,W:int,
                     bp_lo:Optional[float], bp_hi:Optional[float]):
    out_root.mkdir(parents=True, exist_ok=True)
    idx_csv = out_root / "index.csv"
    all_rows=[]
    for dirpath,_,filenames in os.walk(root):
        for fn in filenames:
            if not fn.lower().endswith(".mat"): continue
            src = Path(dirpath)/fn
            try:
                rows = process_one_file(src, out_root, mode, K,R,overlap,Omax,D,H,W,bp_lo,bp_hi)
                all_rows.extend(rows)
                if rows:
                    print(f"✔ {src} -> {sum(r[2] for r in rows)} 片")
            except Exception as e:
                print(f"✖ 失败: {src}\n   {e}")
    with open(idx_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["src_file","var","num_segments","rpm_used","mode","out_path"])
        w.writerows(all_rows)
    print(f"\n✅ 完成：输出索引 {idx_csv}，共 {len(all_rows)} 行")

# -------- CLI ----------
if __name__=="__main__":
    ap = argparse.ArgumentParser(description="遍历 .mat；无 RPM 则估计；仅输出 .npy 张量")
    ap.add_argument("root", type=str, help="源数据根目录")
    ap.add_argument("--out_root", type=str, required=True, help="输出根目录")
    ap.add_argument("--mode", type=str, default="vector", choices=["vector","image"])
    ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--R", type=int, default=32)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--Omax", type=int, default=50)
    ap.add_argument("--D", type=int, default=256)
    ap.add_argument("--H", type=int, default=128)
    ap.add_argument("--W", type=int, default=128)
    ap.add_argument("--bp_lo", type=float, default=500.0, help="设为 -1 关闭")
    ap.add_argument("--bp_hi", type=float, default=10000.0, help="设为 -1 关闭")
    args = ap.parse_args()

    root = Path(args.root); out_root = Path(args.out_root)
    bp_lo = None if (args.bp_lo is not None and args.bp_lo < 0) else args.bp_lo
    bp_hi = None if (args.bp_hi is not None and args.bp_hi < 0) else args.bp_hi

    walk_and_process(root, out_root, args.mode,
                     args.K, args.R, args.overlap, args.Omax, args.D, args.H, args.W,
                     bp_lo, bp_hi)
