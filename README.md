# 小数据集猫狗分类（传统方法 vs 深度学习方法）

第一次作业：分别使用**传统方法**和**深度学习方法**实现小数据集猫狗分类。

数据集：`猫狗数据集100张.zip`（猫 50 张 + 狗 50 张，共 100 张）

## 项目结构

```
cat_dog_classifier/
├── prepare_data.py          # 数据整理与划分（train 80% / val 10% / test 10%）
├── traditional_method.py    # 传统方法：SIFT特征 + 词袋模型(BoVW) + SVM（可选 HOG 特征）
├── deep_learning_method.py  # 深度学习方法：CNN 卷积神经网络（PyTorch）
├── data/                    # 原始图片（cat/、dog/）
├── datasets/                # 划分后的数据集（运行 prepare_data.py 生成）
├── models/                  # 训练好的模型（svm_model.pkl / cnn_model.pth）
├── output/                  # 混淆矩阵图、训练曲线图
├── requirements.txt
└── README.md
```

## 运行步骤

```bash
# 1. 安装依赖（已装好可跳过）
pip install -r requirements.txt

# 2. 整理并划分数据集（一次即可）
python prepare_data.py

# 3. 传统方法：SIFT + 词袋 + SVM
python traditional_method.py

# 4. 深度学习方法：CNN
python deep_learning_method.py
```

## 两种方法对比

| 对比项 | 传统方法 | 深度学习方法 |
|---|---|---|
| 特征 | SIFT 关键点 + 词袋直方图（人工设计特征，也可用 HOG） | CNN 自动学习特征（ResNet18 预训练迁移学习） |
| 分类器 | SVM 支持向量机（RBF 核 + 网格搜索） | ResNet18 微调（也可用 --model cnn 从零训练） |
| 数据增强 | 无 | 随机裁剪、翻转、旋转、色彩抖动 |
| 训练时间 | 约 5 秒 | 约 70 秒（CPU） |
| 测试集准确率 | **80%**（4/5 猫 + 4/5 狗） | **100%**（5/5 猫 + 5/5 狗） |

## 运行结果

测试集共 10 张（5 猫 + 5 狗），两种方法运行结果：

| 方法 | 测试准确率 | 猫正确 | 狗正确 |
|---|---|---|---|
| 传统方法（SIFT + BoVW + SVM） | 80.00% | 4/5 | 4/5 |
| 深度学习方法（ResNet18 迁移学习） | 100.00% | 5/5 | 5/5 |

> 结论：小数据集上手工特征（SIFT）分类能力有限；深度学习借助 ImageNet 预训练特征 + 数据增强，准确率更高。结果图保存在 `output/` 目录（混淆矩阵、训练曲线）。

## Git 管理（作业要求）

```bash
git init
git add .
git commit -m "初始提交：传统方法 + 深度学习方法实现猫狗分类"
# 关联远程仓库并推送
git remote add origin <你的GitHub仓库地址>
git push -u origin master
```

> 说明：`datasets/`、`models/`、`output/` 已在 `.gitignore` 中忽略（可重新生成），只提交代码。
