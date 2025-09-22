import os
import csv
import argparse
import traceback
from typing import List, Tuple

# 这两类文件分别用于 v5/v7 和 v7.3
from scipy.io import whosmat, loadmat
import numpy as np

try:
    import h5py  # v7.3 基于 HDF5
    HAS_H5PY = True
except Exception:
    HAS_H5PY = False


META_KEYS = {"__header__", "__version__", "__globals__"}

def is_mat_file(path: str) -> bool:
    return path.lower().endswith(".mat")

def safe_str(x) -> str:
    try:
        return str(x)
    except Exception:
        return repr(x)

def summarize_np(obj) -> Tuple[str, str, str]:
    """返回 kind, shape, dtype 的字符串表示（scipy.loadmat 加载结果用）"""
    if isinstance(obj, np.ndarray):
        return ("ndarray", safe_str(obj.shape), safe_str(obj.dtype))
    # mat_struct / record 等其他对象
    kind = obj.__class__.__name__
    shape = getattr(obj, "shape", "")
    dtype = getattr(obj, "dtype", "")
    return (kind, safe_str(shape), safe_str(dtype))

def list_vars_v5v7(path: str) -> List[Tuple[str, str, str, str]]:
    """
    通过 scipy.io.whosmat 快速读取 v5/v7 mat 的变量清单。
    返回: (file, var_name, kind, shape/dtype)
    """
    rows = []
    try:
        # whosmat: [(name, shape, matlab_class), ...]
        metas = whosmat(path)
        for name, shape, mclass in metas:
            if name in META_KEYS:
                continue
            # 如果需要更细信息，可 loadmat(squeeze_me=True, struct_as_record=False)
            kind = f"matclass:{mclass}"
            rows.append((path, name, kind, safe_str(shape)))
    except Exception:
        # 某些文件可能不是 v5/v7（比如 v7.3），交给上层处理
        raise
    return rows

def list_vars_v5v7_deep(path: str) -> List[Tuple[str, str, str, str]]:
    """
    可选：深读一次 loadmat，把 dtype/shape 也带上（会较慢）
    """
    rows = []
    data = loadmat(path, squeeze_me=False, struct_as_record=False)
    for k, v in data.items():
        if k in META_KEYS:
            continue
        kind, shape, dtype = summarize_np(v)
        rows.append((path, k, kind, f"{shape}; dtype={dtype}"))
    return rows

def list_vars_v73(path: str) -> List[Tuple[str, str, str, str]]:
    """
    遍历 v7.3 (HDF5) .mat 的所有数据集/组，输出“/组/变量”路径及形状/类型
    """
    rows = []
    if not HAS_H5PY:
        return rows

    def visitor(name, obj):
        # name: HDF5 路径（/group/sub/...）
        if isinstance(obj, h5py.Dataset):
            shape = safe_str(obj.shape)
            dtype = safe_str(obj.dtype)
            rows.append((path, name.lstrip("/"), "dataset", f"{shape}; dtype={dtype}"))
        elif isinstance(obj, h5py.Group):
            # 组本身也可以记录（可选）
            pass

    with h5py.File(path, "r") as f:
        f.visititems(visitor)
    return rows

def detect_and_list_vars(path: str) -> List[Tuple[str, str, str, str]]:
    """
    自动判断 .mat 版本并列出变量：
    - 先尝试 v5/v7 的 whosmat（快）；
    - 失败则尝试 v7.3 的 HDF5 遍历；
    - 若还要更详细信息，可切换为 list_vars_v5v7_deep（慢）。
    """
    # 首先尝试当作 v5/v7
    try:
        rows = list_vars_v5v7(path)
        # 如果想更详细(含 dtype)，用下面这行替换：
        # rows = list_vars_v5v7_deep(path)
        if rows:
            return rows
    except Exception:
        pass

    # 尝试 v7.3 (HDF5)
    try:
        rows = list_vars_v73(path)
        if rows:
            return rows
    except Exception:
        pass

    # 两种都失败，则返回一条错误记录
    return [(path, "<UNREADABLE>", "error", "not v5/v7 or v7.3, or file corrupted")]

def walk_and_dump(root: str, out_csv: str):
    records = []
    for dirpath, dirnames, filenames in os.walk(root):
        for fn in filenames:
            if not is_mat_file(fn):
                continue
            fpath = os.path.join(dirpath, fn)
            try:
                rows = detect_and_list_vars(fpath)
                records.extend(rows)
                # 控制台简要输出
                print(f"✔ {fpath}")
                for _, vname, kind, meta in rows[:10]:
                    print(f"   - {vname} [{kind}] {meta}")
                if len(rows) > 10:
                    print(f"   ... 共 {len(rows)} 条")
            except Exception:
                print(f"✖ 读取失败: {fpath}")
                traceback.print_exc()

    # 写 CSV
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "var", "kind", "shape_or_meta"])
        for rec in records:
            w.writerow(rec)
    print(f"\n✅ 已写出 {len(records)} 条变量记录到: {out_csv}")

def main():
    ap = argparse.ArgumentParser(description="遍历 .mat 文件并列出变量清单（兼容 v5/v7 与 v7.3）")
    ap.add_argument("root", help="数据集根目录")
    ap.add_argument("-o", "--out", default="mat_variables.csv", help="输出 CSV 路径")
    args = ap.parse_args()
    walk_and_dump(args.root, args.out)

if __name__ == "__main__":
    main()
