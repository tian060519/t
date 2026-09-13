# -*- coding: utf-8 -*-
"""
traditional_method.py
传统方法实现猫狗分类：SIFT 特征 + 词袋模型(BoVW) + SVM 支持向量机。

流程：
    1. 读取图像并统一缩放
    2. 提取 SIFT 关键点描述子（可选 --feature hog 用 HOG 特征）
    3. 在训练集上用 K-Means 构建视觉词表，把每张图编码为词袋直方图
    4. 特征标准化后训练 SVM（RBF 核 + 网格搜索选参）
    5. 在测试集上评估并输出准确率、分类报告、混淆矩阵
    6. 保存模型（SVM + KMeans + Scaler 打包）

运行方式：
    python traditional_method.py                 # 默认 SIFT + BoVW
    python traditional_method.py --feature hog   # 可选 HOG 特征
"""
import argparse
import time
import numpy as np
import cv2
from pathlib import Path
from sklearn.svm import SVC
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 中文字体（Windows）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---------- 配置 ----------
ROOT = Path(r"D:\cxdownload\cat_dog_classifier")
DATA_DIR = ROOT / "datasets"
CLASSES = ["cat", "dog"]
SIFT_IMG_SIZE = 256      # SIFT 用较大分辨率以提取更多关键点
HOG_IMG_SIZE = 128
N_CLUSTERS = 50          # 词袋词典大小
OUTPUT_DIR = ROOT / "output"
MODEL_PATH = ROOT / "models" / "svm_model.pkl"
# -------------------------


def load_images(split: str, img_size: int):
    """读取某个划分（train/val/test）下所有图片和标签。"""
    X, y = [], []
    for idx, c in enumerate(CLASSES):
        folder = DATA_DIR / split / c
        for img_path in sorted(folder.glob("*.jpg")):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img = cv2.resize(img, (img_size, img_size))
            X.append(img)
            y.append(idx)
    return X, y


def extract_sift_descriptors(img):
    """提取 SIFT 描述子。"""
    sift = cv2.SIFT_create()
    _, desc = sift.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)
    return desc  # 可能为 None（无关键点）


def extract_hog_feature(img):
    """提取 HOG 特征（备选方案）。"""
    from skimage.feature import hog as sk_hog
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    feat = sk_hog(
        gray, orientations=9, pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), channel_axis=None,
    ).astype(np.float64)
    # 补充 HSV 颜色直方图
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = np.concatenate([
        cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten(),
        cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten(),
        cv2.calcHist([hsv], [2], None, [32], [0, 256]).flatten(),
    ])
    hist = hist / (hist.sum() + 1e-6)
    return np.concatenate([feat, hist])


def build_bovw(train_imgs, n_clusters):
    """在训练集描述子上用 K-Means 构建视觉词表。"""
    print(f"   提取 SIFT 描述子（训练集 {len(train_imgs)} 张）...")
    all_desc = []
    desc_list = []
    for img in train_imgs:
        d = extract_sift_descriptors(img)
        desc_list.append(d)
        if d is not None:
            all_desc.append(d)
    all_desc = np.vstack(all_desc)
    print(f"   总描述子数: {len(all_desc)}，聚类为 {n_clusters} 个视觉词")
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=512)
    kmeans.fit(all_desc)

    def encode(desc):
        if desc is None or len(desc) == 0:
            return np.zeros(n_clusters)
        labels = kmeans.predict(desc)
        hist = np.bincount(labels, minlength=n_clusters).astype(np.float64)
        return hist / (hist.sum() + 1e-6)

    feats = np.array([encode(d) for d in desc_list])
    return kmeans, feats


def encode_with_kmeans(kmeans, imgs, n_clusters):
    """用训练好的词表编码新图片。"""
    feats = []
    for img in imgs:
        d = extract_sift_descriptors(img)
        if d is None or len(d) == 0:
            feats.append(np.zeros(n_clusters))
        else:
            labels = kmeans.predict(d)
            hist = np.bincount(labels, minlength=n_clusters).astype(np.float64)
            feats.append(hist / (hist.sum() + 1e-6))
    return np.array(feats)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature", choices=["sift", "hog"], default="sift",
                        help="特征类型：sift= SIFT+BoVW（默认），hog= HOG+颜色直方图")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / "models").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"===== 传统方法: {args.feature.upper()} 特征 + SVM =====")

    img_size = SIFT_IMG_SIZE if args.feature == "sift" else HOG_IMG_SIZE

    print("1. 读取图片...")
    X_train, y_train = load_images("train", img_size)
    X_val, y_val = load_images("val", img_size)
    X_test, y_test = load_images("test", img_size)
    print(f"   训练集: {len(X_train)}  验证集: {len(X_val)}  测试集: {len(X_test)}")

    print("2. 提取特征...")
    if args.feature == "sift":
        kmeans, X_train_feat = build_bovw(X_train, N_CLUSTERS)
        X_val_feat = encode_with_kmeans(kmeans, X_val, N_CLUSTERS)
        X_test_feat = encode_with_kmeans(kmeans, X_test, N_CLUSTERS)
    else:
        kmeans = None
        X_train_feat = np.array([extract_hog_feature(img) for img in X_train])
        X_val_feat = np.array([extract_hog_feature(img) for img in X_val])
        X_test_feat = np.array([extract_hog_feature(img) for img in X_test])
    y_train, y_val, y_test = np.array(y_train), np.array(y_val), np.array(y_test)
    print(f"   特征维度: {X_train_feat.shape[1]}")

    print("3. 特征标准化...")
    scaler = StandardScaler()
    X_train_feat = scaler.fit_transform(X_train_feat)
    X_val_feat = scaler.transform(X_val_feat)
    X_test_feat = scaler.transform(X_test_feat)

    print("4. 训练 SVM（RBF 核 + 网格搜索选参）...")
    param_grid = {"C": [0.1, 1, 10], "gamma": ["scale", 0.01, 0.001]}
    svm = SVC(kernel="rbf", class_weight="balanced")
    grid = GridSearchCV(svm, param_grid, cv=3, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train_feat, y_train)
    best = grid.best_estimator_
    print(f"   最优参数: {grid.best_params_}")
    print(f"   验证集准确率: {accuracy_score(y_val, best.predict(X_val_feat)):.4f}")

    print("5. 测试集评估...")
    y_pred = best.predict(X_test_feat)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n========== 传统方法（{args.feature.upper()} + SVM）测试结果 ==========")
    print(f"测试集准确率: {acc:.4f} ({acc * 100:.2f}%)")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=CLASSES, digits=4))
    print("混淆矩阵:")
    print(confusion_matrix(y_test, y_pred))

    # 保存模型
    joblib.dump({"svm": best, "kmeans": kmeans, "scaler": scaler, "feature": args.feature}, MODEL_PATH)
    print(f"\n模型已保存: {MODEL_PATH}")
    print(f"总耗时: {time.time() - t0:.1f}s")

    # 混淆矩阵图
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    tick = np.arange(len(CLASSES))
    plt.xticks(tick, CLASSES)
    plt.yticks(tick, CLASSES)
    plt.xlabel("预测类别")
    plt.ylabel("真实类别")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.title(f"传统方法混淆矩阵 (准确率 {acc:.2%})")
    plt.tight_layout()
    cm_path = OUTPUT_DIR / "traditional_confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    print(f"混淆矩阵图已保存: {cm_path}")


if __name__ == "__main__":
    main()
