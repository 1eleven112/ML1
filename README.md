# Transformer模型压缩研究

## 项目简介

本项目专注于Transformer模型压缩技术的研究与实现，包括模型剪枝、量化、知识蒸馏等多种压缩方法。

## 目录结构

```
ML1/
├── README.md                    # 项目说明文档
├── docs/                        # 详细文档
│   ├── background.md           # 问题背景与动机
│   ├── innovation.md           # 创新与网络结构
│   └── results.md              # 实验结果
├── src/                        # 源代码
│   ├── models/                 # 模型定义
│   ├── compression/            # 压缩技术实现
│   ├── training/               # 训练脚本
│   └── utils/                  # 工具函数
├── experiments/                # 实验脚本和结果
├── notebooks/                  # Jupyter notebooks示例
├── requirements.txt            # 依赖包
└── setup.py                    # 安装配置
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行示例

```bash
# 训练基础Transformer模型
python src/training/train_base.py

# 应用模型压缩
python src/training/train_compressed.py --method pruning

# 运行实验
python experiments/run_experiments.py
```

## 主要特性

- **多种压缩技术**: 实现了剪枝、量化、知识蒸馏等多种压缩方法
- **完整的实验框架**: 包含训练、评估、可视化的完整流程
- **详细的文档**: 包含理论背景、技术创新和实验分析

## 研究背景

详见 [docs/background.md](docs/background.md)

## 技术创新

详见 [docs/innovation.md](docs/innovation.md)

## 实验结果

详见 [docs/results.md](docs/results.md)

## 参考文献

1. Vaswani et al., "Attention is All You Need", NeurIPS 2017
2. Michel et al., "Are Sixteen Heads Really Better than One?", NeurIPS 2019
3. Sanh et al., "DistilBERT, a distilled version of BERT", arXiv 2019
4. Fan et al., "Reducing Transformer Depth on Demand with Structured Dropout", ICLR 2020

## 作者

研究方向：Transformer模型压缩