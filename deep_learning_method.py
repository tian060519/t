# -*- coding: utf-8 -*-
"""
deep_learning_method.py
深度学习方法实现猫狗分类（PyTorch）。

默认方案：ResNet18 迁移学习（ImageNet 预训练 + 微调）——小数据集标准做法。
备选方案：从零训练小型 CNN（--model cnn）。

流程：
    1. 使用 torchvision 读取数据，训练集做数据增强（随机裁剪/翻转/旋转/色彩抖动）
    2. 加载 ImageNet 预训练 ResNet18，冻结主干，替换分类头并微调最后一层卷积块
    3. 训练若干轮，保存验证集上最优模型
    4. 在测试集上评估，输出准确率、分类报告、混淆矩阵
    5. 绘制训练/验证损失与准确率曲线

运行方式：
    python deep_learning_method.py             # 默认 ResNet18 迁移学习
    python deep_learning_method.py --model cnn # 可选从零训练小型 CNN
"""
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# 中文字体（Windows）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---------- 配置 ----------
ROOT = Path(r"D:\cxdownload\cat_dog_classifier")
DATA_DIR = ROOT / "datasets"
OUTPUT_DIR = ROOT / "output"
MODEL_PATH = ROOT / "models" / "cnn_model.pth"
IMG_SIZE = 224          # 迁移学习用 224x224
BATCH_SIZE = 16
EPOCHS = 30             # 迁移学习 30 轮足够
LEARNING_RATE = 1e-3
SEED = 42
CLASSES = ["cat", "dog"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -------------------------

torch.manual_seed(SEED)
np.random.seed(SEED)

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


# ---------- 数据增强与加载 ----------
def get_transform(augment: bool):
    if augment:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


# ---------- 从零训练的小型 CNN（备选） ----------
class SimpleCNN(nn.Module):
    """小型 CNN：3 个卷积块 + 全局平均池化 + 全连接输出 2 类。"""
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(model_name: str):
    """构建模型：resnet18=迁移学习；cnn=从零训练。"""
    if model_name == "resnet18":
        try:
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        except Exception as e:
            raise RuntimeError(
                f"预训练权重下载失败（需要联网）: {e}\n"
                f"可改用 --model cnn 从零训练，或检查网络后重试。"
            )
        # 冻结主干，替换分类头，微调最后一个卷积块
        for p in model.parameters():
            p.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, 2)
        for p in model.layer4.parameters():
            p.requires_grad = True
        return model
    return SimpleCNN(num_classes=2)


# ---------- 训练 ----------
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    criterion = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = criterion(out, y)
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["resnet18", "cnn"], default="resnet18",
                        help="模型：resnet18=迁移学习（默认），cnn=从零训练小型CNN")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="训练轮数")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / "models").mkdir(parents=True, exist_ok=True)
    print(f"设备: {DEVICE}")
    t0 = time.time()
    model_name = args.model
    print(f"===== 深度学习方法: {model_name} =====")

    # 1. 数据加载
    print("1. 加载数据...")
    train_ds = datasets.ImageFolder(DATA_DIR / "train", transform=get_transform(augment=True))
    val_ds = datasets.ImageFolder(DATA_DIR / "val", transform=get_transform(augment=False))
    test_ds = datasets.ImageFolder(DATA_DIR / "test", transform=get_transform(augment=False))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"   训练集: {len(train_ds)}  验证集: {len(val_ds)}  测试集: {len(test_ds)}")

    # 2. 构建模型
    print("2. 构建模型...")
    model = build_model(model_name).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    if model_name == "resnet18":
        # 只优化需要梯度的参数（分类头 + layer4）
        optimizer = optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=LEARNING_RATE, weight_decay=1e-4)
    else:
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    print(model)

    # 3. 训练
    print(f"3. 训练 {args.epochs} 轮...")
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        va_loss, va_acc = evaluate(model, val_loader)
        scheduler.step()
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), MODEL_PATH)
        if epoch % 5 == 0 or epoch == 1:
            print(f"   epoch {epoch:2d}/{args.epochs} | "
                  f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
                  f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}")

    # 4. 加载最优模型，测试集评估
    print("4. 测试集评估...")
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            out = model(x.to(DEVICE))
            all_preds.extend(out.argmax(1).cpu().tolist())
            all_labels.extend(y.tolist())
    acc = np.mean(np.array(all_preds) == np.array(all_labels))

    print(f"\n========== 深度学习方法（{model_name}）测试结果 ==========")
    print(f"测试集准确率: {acc:.4f} ({acc * 100:.2f}%)")
    print("\n分类报告:")
    print(classification_report(all_labels, all_preds, target_names=CLASSES, digits=4))
    print("混淆矩阵:")
    print(confusion_matrix(all_labels, all_preds))
    print(f"\n最优验证准确率: {best_acc:.4f}")
    print(f"模型已保存: {MODEL_PATH}")
    print(f"总耗时: {time.time() - t0:.1f}s")

    # 5. 画训练曲线
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.suptitle("CNN 训练曲线")
    plt.tight_layout()
    curve_path = OUTPUT_DIR / "cnn_training_curves.png"
    plt.savefig(curve_path, dpi=150)
    print(f"训练曲线已保存: {curve_path}")

    # 6. 画混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
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
    plt.title(f"CNN 混淆矩阵 (准确率 {acc:.2%})")
    plt.tight_layout()
    cm_path = OUTPUT_DIR / "cnn_confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    print(f"混淆矩阵图已保存: {cm_path}")


if __name__ == "__main__":
    main()
