import os, re, csv, argparse
from pathlib import Path
import numpy as np

LABEL_KEYS = ["DE_B","DE_IR","DE_OR","FE_B","FE_IR","FE_OR"]

# -------------------- 路径规范化与提取 --------------------
def norm_rel(p: str) -> str:
    return p.replace("\\","/").strip().lower()

def strip_drive_and_lead(p: str) -> str:
    p = re.sub(r'^[a-z]:', '', p, flags=re.IGNORECASE)  # 去掉盘符
    return p.lstrip("/")

def tail_from_markers(p: str):
    """提取以数据集关键目录开头的尾部路径，用于跨盘符/不同根目录时对齐"""
    pn = norm_rel(p)
    pn = strip_drive_and_lead(pn)
    for mk in ("de/","fe/","ba/","48khz normal/"):
        i = pn.find(mk)
        if i >= 0:
            return pn[i:]
    return pn

# -------------------- 读取标签 CSV --------------------
def load_labels(labels_csv: Path):
    """
    返回多键字典：各种规范化后的 path key -> 6维 one-hot
    兼容:
      path, DE_B, DE_IR, DE_OR, FE_B, FE_IR, FE_OR
    或:
      path, class_id / label
    """
    lab = {}
    with open(labels_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        hdr = [h.strip() for h in reader.fieldnames]
        has_onehot = all(k in hdr for k in LABEL_KEYS)
        has_classid = ("class_id" in hdr) or ("label" in hdr)
        path_key = "path" if "path" in hdr else hdr[0]

        for row in reader:
            raw_path = row[path_key]
            if has_onehot:
                y = [int(float(row[k])) for k in LABEL_KEYS]
            elif has_classid:
                cid = int(row["class_id"] if "class_id" in row else row["label"])
                y = [0]*6
                if 0 <= cid < 6: y[cid] = 1
            else:
                raise ValueError("labels_csv 需包含六列标签或 class_id。")

            # 生成多种候选 key
            keys = set()
            p1 = norm_rel(raw_path)                # 原样规范化
            p2 = strip_drive_and_lead(p1)          # 去盘符
            p3 = tail_from_markers(raw_path)       # 从数据集关键目录起
            keys.update([p1, p2, p3])

            # 仅文件名、最后两级路径也加入候选
            parts = p2.split("/")
            if parts:
                keys.add(parts[-1])                # 仅文件名
            if len(parts) >= 2:
                keys.add("/".join(parts[-2:]))

            for k in keys:
                lab[k] = y
    return lab

# -------------------- 读取 index.csv（权威映射） --------------------
def load_index_map(index_csv: Path, npy_root: Path):
    """
    返回 dict: 多种规范化的 npy 路径 key -> 对应 .mat 的多种 key（同样规范化）
    期望列名包含: out_path, src_file
    """
    idx = {}
    with open(index_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        hdr = [h.strip().lower() for h in reader.fieldnames]
        # 自动识别列名
        def find_col(candidates):
            for c in candidates:
                if c in hdr: return c
            return None

        col_out = find_col(["out_path","out","npy","tensor_path"])
        col_src = find_col(["src_file","src","mat","mat_path"])
        if not col_out or not col_src:
            raise ValueError(f"index.csv 里找不到必要列 out_path/src_file，实际列: {reader.fieldnames}")

        for row in reader:
            npy_path_raw = row[col_out]
            mat_path_raw = row[col_src]

            # 为 npy 生成多种 key
            npy_keys = set()
            npy_abs = norm_rel(npy_path_raw)
            npy_keys.add(npy_abs)
            try:
                # 相对 npy_root 的路径
                npy_rel = Path(npy_path_raw)
                if npy_rel.is_absolute():
                    npy_rel = npy_rel.relative_to(npy_root)
                else:
                    npy_rel = Path(npy_root, npy_rel).relative_to(npy_root)
                npy_keys.add(norm_rel(npy_rel.as_posix()))
            except Exception:
                pass
            npy_keys.add(norm_rel(Path(npy_path_raw).name))        # 仅文件名
            npy_keys.add(tail_from_markers(npy_path_raw))          # 从关键目录起

            # 为 mat 生成多种 key
            mat_keys = set()
            p1 = norm_rel(mat_path_raw)
            p2 = strip_drive_and_lead(p1)
            p3 = tail_from_markers(mat_path_raw)
            mat_keys.update([p1, p2, p3])
            parts = p2.split("/")
            if parts:
                mat_keys.add(parts[-1])
            if len(parts) >= 2:
                mat_keys.add("/".join(parts[-2:]))

            for nk in npy_keys:
                idx[nk] = mat_keys  # 记录到“可能的 mat key 集合”
    return idx

# -------------------- 旧的回退：从 .npy 名字反推 .mat --------------------
def guess_mat_rel_from_npy(npy_path: Path, npy_root: Path):
    rel_dir = npy_path.parent.relative_to(npy_root).as_posix()
    base = npy_path.stem
    base = re.sub(r'_(vecs|imgs)$', '', base, flags=re.IGNORECASE)
    mat_stem = re.sub(r'_X\d+_(DE|FE|BA)_time$', '', base, flags=re.IGNORECASE)
    if mat_stem == base:
        mat_stem = re.sub(r'_(DE|FE|BA)(_time)?$', '', base, flags=re.IGNORECASE)
    rel_mat = f"{rel_dir}/{mat_stem}.mat" if rel_dir else f"{mat_stem}.mat"
    rel_tail = tail_from_markers(rel_mat)
    # 返回两种 key 便于匹配
    return [norm_rel(rel_mat), rel_tail, norm_rel(Path(rel_mat).name)]

# -------------------- 主流程 --------------------
def main(npy_root: Path, labels_csv: Path, out_root: Path, index_csv: Path = None):
    labels = load_labels(labels_csv)
    index_map = load_index_map(index_csv, npy_root) if index_csv else {}

    out_root.mkdir(parents=True, exist_ok=True)
    patterns = ["**/*_vecs.npy", "**/*_imgs.npy"]
    npy_list = []
    for pat in patterns:
        npy_list.extend(npy_root.glob(pat))
    npy_list = sorted(npy_list)

    manifest_rows = []
    missing = []
    total_files = 0
    total_samples = 0

    for npy_path in npy_list:
        # 1) 先用 index.csv 的权威映射
        mat_keys_candidates = None
        npy_keys_try = {
            norm_rel(str(npy_path)),
            norm_rel(npy_path.name),
            norm_rel(npy_path.parent.relative_to(npy_root).as_posix() + "/" + npy_path.name),
            tail_from_markers(str(npy_path)),
        }
        # 相对 npy_root 的规范化路径
        npy_rel_key = norm_rel(npy_path.relative_to(npy_root).as_posix())
        npy_keys_try.add(npy_rel_key)

        for nk in list(npy_keys_try):
            if nk in index_map:
                mat_keys_candidates = index_map[nk]
                break

        # 2) 若 index 没找到，再用“猜测”法
        if not mat_keys_candidates:
            mat_keys_candidates = set(guess_mat_rel_from_npy(npy_path, npy_root))

        # 3) 用 labels 字典去匹配这些 mat key
        y_row = None
        chosen_key = None
        for mk in mat_keys_candidates:
            if mk in labels:
                y_row = np.array(labels[mk], dtype=np.int64)
                chosen_key = mk
                break

        if y_row is None:
            # 再试一轮：把候选 key 各种衍生一下匹配
            extra_try = set()
            for mk in list(mat_keys_candidates):
                extra_try.update([
                    norm_rel(mk),
                    strip_drive_and_lead(mk),
                    tail_from_markers(mk),
                    norm_rel(Path(mk).name),
                ])
            for mk in extra_try:
                if mk in labels:
                    y_row = np.array(labels[mk], dtype=np.int64)
                    chosen_key = mk
                    break

        if y_row is None:
            print(f"[WARN] 标签缺失(已用 index 优先): {npy_path} -> {list(mat_keys_candidates)[:1]}")
            missing.append([str(npy_path), ";".join(sorted(mat_keys_candidates))])
            continue

        X = np.load(npy_path)    # [N, D] 或 [N, H, W]
        N = X.shape[0]
        y = np.tile(y_row, (N, 1))

        out_dir = out_root / npy_path.parent.relative_to(npy_root)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_npz = out_dir / (npy_path.stem + "_paired.npz")
        np.savez_compressed(out_npz, X=X, y=y, src_mat_key=chosen_key, src_npy=str(npy_path))

        total_files += 1
        total_samples += N
        manifest_rows.append([str(npy_path), str(out_npz), chosen_key, N])
        print(f"✔ {npy_path} -> {out_npz}  (N={N})")

    # 写清单
    man = out_root / "pairs_manifest.csv"
    with open(man, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["src_npy", "out_npz", "matched_mat_key", "num_vectors"])
        w.writerows(manifest_rows)

    if missing:
        miss = out_root / "missing_labels.csv"
        with open(miss, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["src_npy", "candidate_mat_keys"])
            w.writerows(missing)

    print(f"\n✅ 完成：配对文件 {total_files} 个，总向量 {total_samples} 个")
    if missing:
        print(f"⚠ 仍有 {len(missing)} 个 .npy 未匹配到标签；已写入 {miss}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="将每个 .npy 张量与标签配对（优先使用 index.csv 映射），输出 .npz")
    ap.add_argument("--npy_root", required=True, help="张量 .npy 根目录（含 *_vecs.npy / *_imgs.npy）")
    ap.add_argument("--labels_csv", required=True, help="标签 CSV：path + 6 列 one-hot 或 class_id/label")
    ap.add_argument("--out_root", required=True, help="输出根目录（写 .npz 与 manifest）")
    ap.add_argument("--index_csv", required=False, help="切片阶段生成的 index.csv（含 out_path 与 src_file）")
    args = ap.parse_args()

    npy_root = Path(args.npy_root)
    labels_csv = Path(args.labels_csv)
    out_root = Path(args.out_root)
    index_csv = Path(args.index_csv) if args.index_csv else None

    main(npy_root, labels_csv, out_root, index_csv)
