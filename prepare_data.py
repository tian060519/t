# -*- coding: utf-8 -*-
"""
prepare_data.py
数据整理与划分：把原始 cat/dog 图片整理成 PyTorch ImageFolder / sklearn 可用的目录结构。
最终目录结构：
    datasets/
        train/cat  train/dog
        val/cat    val/dog
        test/cat   test/dog
划分比例：训练 80% / 验证 10% / 测试 10%（每类内部随机划分，随机种子固定保证可复现）。
"""
import os
import shutil
import random
from pathlib import Path

# ---------- 配置 ----------
SRC_DIR = Path(r"D:\cxdownload\cat_dog_dataset")     # 原始图片目录（100 张）
ROOT = Path(r"D:\cxdownload\cat_dog_classifier")     # 项目根目录
DATA_DIR = ROOT / "datasets"                          # 划分后的数据集
RATIO = (0.8, 0.1, 0.1)                               # train / val / test
SEED = 42
# -------------------------


def main():
    random.seed(SEED)
    classes = ["cat", "dog"]

    # 1. 收集每类图片
    class_files = {}
    for c in classes:
        files = sorted(SRC_DIR.glob(f"{c}.*"))
        class_files[c] = files
        print(f"[{c}] 共 {len(files)} 张")

    # 2. 按比例划分
    split_map = {"train": [], "val": [], "test": []}
    for c, files in class_files.items():
        random.shuffle(files)
        n = len(files)
        n_train = int(n * RATIO[0])
        n_val = int(n * RATIO[1])
        split_map["train"] += [(f, c) for f in files[:n_train]]
        split_map["val"] += [(f, c) for f in files[n_train:n_train + n_val]]
        split_map["test"] += [(f, c) for f in files[n_train + n_val:]]

    # 3. 复制到目标目录
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    for split in split_map:
        for f, c in split_map[split]:
            dst_dir = DATA_DIR / split / c
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst_dir / f.name)

    # 4. 打印统计
    print("\n===== 划分结果 =====")
    for split, items in split_map.items():
        n_cat = sum(1 for _, c in items if c == "cat")
        n_dog = sum(1 for _, c in items if c == "dog")
        print(f"{split:5s}: cat={n_cat:2d}  dog={n_dog:2d}  合计={len(items):3d}")
    print(f"\n数据集已保存到: {DATA_DIR}")


if __name__ == "__main__":
    main()
