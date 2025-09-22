# make_labels.py
import argparse, csv, re
from pathlib import Path

# 标签顺序: [DE_B, DE_IR, DE_OR, FE_B, FE_IR, FE_OR]
LABELS = ["DE_B","DE_IR","DE_OR","FE_B","FE_IR","FE_OR"]

def label_from_path(p: Path):
    parts = [s.lower() for s in p.parts]
    # Normal：全零
    if any("normal" in s for s in parts):
        return [0,0,0,0,0,0]

    # 是否在 DE 或 FE 分支
    is_de = any(re.fullmatch(r"de", s) for s in parts)
    is_fe = any(re.fullmatch(r"fe", s) for s in parts)

    # 在 DE/FE 下的“类别”目录名可能是 "12kHz B", "48kHz IR", "12kHz OR" 等
    def has_token(token):
        pat = re.compile(rf"(?:^|[^A-Za-z]){token}(?:[^A-Za-z]|$)", re.IGNORECASE)
        return any(pat.search(seg) for seg in parts)

    if is_de:
        if has_token("IR"):
            return [0,1,0,0,0,0]
        if has_token("OR"):
            return [0,0,1,0,0,0]
        if has_token("B"):
            return [1,0,0,0,0,0]
    if is_fe:
        if has_token("IR"):
            return [0,0,0,0,1,0]
        if has_token("OR"):
            return [0,0,0,0,0,1]
        if has_token("B"):
            return [0,0,0,1,0,0]

    # 没匹配到就标记未知（全零）并提示
    return None

def main(root, out_csv):
    root = Path(root)
    rows = []
    for mat in root.rglob("*.mat"):
        y = label_from_path(mat)
        if y is None:
            print(f"[WARN] 未能从路径推断标签: {mat}")
            continue
        rel = mat.relative_to(root)
        rows.append([str(rel).replace("\\","/")] + y)

    rows.sort(key=lambda r: r[0])
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path"] + LABELS)
        w.writerows(rows)

    print(f"✅ 已写出 {len(rows)} 条到 {out_csv}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="数据根目录")
    ap.add_argument("-o","--out", default="labels.csv", help="输出 CSV 路径")
    args = ap.parse_args()
    main(args.root, args.out)
