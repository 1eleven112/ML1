# Transformer模型压缩 - 使用指南

## 目录
1. [快速开始](#快速开始)
2. [项目结构](#项目结构)
3. [核心功能](#核心功能)
4. [使用示例](#使用示例)
5. [实验复现](#实验复现)
6. [API文档](#api文档)

## 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/1eleven112/ML1.git
cd ML1

# 安装依赖
pip install -r requirements.txt

# 或者使用setup.py安装
pip install -e .
```

### 5分钟教程

```python
import torch
from src.models.transformer import create_base_model
from src.compression.techniques import create_compressed_model

# 1. 创建基础模型
model = create_base_model(vocab_size=10000, num_classes=2)

# 2. 配置压缩
config = {
    'prune_layers': [4, 5],    # 剪枝第4、5层
    'quantize': True,           # 启用量化
    'quantize_bits': 8          # 8-bit量化
}

# 3. 创建压缩模型
compressed = create_compressed_model(model, config)

# 完成！模型已压缩
```

## 项目结构

```
ML1/
├── docs/                          # 详细文档
│   ├── background.md             # 研究背景与动机
│   ├── innovation.md             # 技术创新与网络结构
│   └── results.md                # 实验结果分析
│
├── src/                          # 源代码
│   ├── models/                   # 模型定义
│   │   └── transformer.py        # Transformer实现
│   ├── compression/              # 压缩技术
│   │   └── techniques.py         # 剪枝、量化、蒸馏
│   ├── training/                 # 训练脚本
│   │   ├── train_base.py        # 训练基础模型
│   │   └── train_compressed.py  # 训练压缩模型
│   └── utils/                    # 工具函数
│       └── helpers.py            # 辅助函数
│
├── experiments/                  # 实验脚本
│   └── run_experiments.py       # 完整实验套件
│
├── notebooks/                    # Jupyter示例
│   └── compression_demo.ipynb   # 交互式演示
│
├── requirements.txt              # 依赖列表
├── setup.py                      # 安装配置
└── README.md                     # 项目说明
```

## 核心功能

### 1. 多种压缩技术

#### 模型剪枝
```python
from src.compression.techniques import ModelPruner

pruner = ModelPruner(model)
pruner.prune_layers([4, 5])  # 剪枝指定层
```

#### 量化
```python
from src.compression.techniques import ModelQuantizer

quantizer = ModelQuantizer(model, bits=8)
quantizer.quantize_model()
```

#### 知识蒸馏
```python
from src.compression.techniques import KnowledgeDistiller

distiller = KnowledgeDistiller(
    teacher_model=teacher,
    student_model=student,
    temperature=4.0
)
```

### 2. 完整的训练流程

#### 训练基础模型
```bash
python src/training/train_base.py \
    --num_samples 5000 \
    --num_epochs 10 \
    --batch_size 32
```

#### 训练压缩模型
```bash
python src/training/train_compressed.py \
    --teacher_model_path checkpoints/final_base_model.pt \
    --method hybrid \
    --use_distillation \
    --quantize
```

### 3. 模型评估

```python
from src.utils.helpers import evaluate_model, measure_inference_time

# 评估准确率
metrics = evaluate_model(model, test_loader, device)
print(f"准确率: {metrics['accuracy']:.2f}%")

# 测量推理速度
avg_time, std_time = measure_inference_time(model, test_loader, device)
print(f"推理时间: {avg_time:.2f}ms")
```

### 4. 使用预训练模型

#### 方法1: 使用命令行参数

```bash
# 从预训练模型继续训练
python src/training/train_base.py \
    --pretrained_model checkpoints/best_base_model.pt \
    --num_epochs 5 \
    --learning_rate 1e-5

# 使用预训练模型作为教师模型进行压缩
python src/training/train_compressed.py \
    --teacher_model_path checkpoints/best_base_model.pt \
    --use_distillation
```

#### 方法2: 使用Python API

```python
from src.models.transformer import load_pretrained_model

# 加载预训练模型
model = load_pretrained_model(
    model_path='checkpoints/best_base_model.pt',
    vocab_size=10000,
    num_classes=2,
    device='cuda'
)

# 直接用于推理
model.eval()
with torch.no_grad():
    predictions = model(input_tensor)

# 或继续微调
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
# ... 训练代码 ...
```

#### 支持的模型格式

本框架支持多种预训练模型格式：

```python
# 格式1: 完整checkpoint（推荐）
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'loss': loss
}
torch.save(checkpoint, 'model.pt')

# 格式2: 仅state_dict
torch.save(model.state_dict(), 'model.pt')

# 格式3: 使用我们的save_checkpoint函数
from src.utils.helpers import save_checkpoint
save_checkpoint(model, optimizer, epoch, loss, 'model.pt')
```

## 使用示例

### 示例1: 基础使用

```python
import torch
from torch.utils.data import DataLoader
from src.models.transformer import create_base_model
from src.utils.helpers import create_dummy_dataset, SimpleTokenizer, TextDataset

# 准备数据
texts, labels = create_dummy_dataset(num_samples=1000)
tokenizer = SimpleTokenizer(vocab_size=5000)
tokenizer.build_vocab(texts)

dataset = TextDataset(texts, labels, tokenizer)
loader = DataLoader(dataset, batch_size=32)

# 创建模型
model = create_base_model(len(tokenizer.word2idx), num_classes=2)

# 训练（这里简化了训练循环）
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

model.train()
for inputs, labels in loader:
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### 示例2: 完整压缩流程

```python
from src.compression.techniques import (
    HeadImportanceCalculator,
    ModelPruner,
    KnowledgeDistiller,
    ModelQuantizer
)

# 1. 训练教师模型
teacher = create_base_model(vocab_size, num_classes=2)
# ... 训练代码 ...

# 2. 计算注意力头重要性
importance_calc = HeadImportanceCalculator(teacher)
importance_scores = importance_calc.compute_head_importance(train_loader, device)

# 3. 剪枝
pruner = ModelPruner(teacher)
pruner.prune_attention_heads(importance_scores, prune_ratio=0.5)
pruner.prune_layers([4, 5])

# 4. 知识蒸馏
student = pruner.model
distiller = KnowledgeDistiller(teacher, student)

for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        distiller.train_step(inputs, labels, optimizer, device)

# 5. 量化
quantizer = ModelQuantizer(student, bits=8)
quantizer.quantize_model()

# 6. 保存
torch.save(student.state_dict(), 'compressed_model.pt')
```

### 示例3: 对比不同压缩配置

```python
configs = [
    {'prune_layers': [5], 'quantize': False},           # 轻度
    {'prune_layers': [4, 5], 'quantize': True},        # 中度
    {'prune_layers': [3, 4, 5], 'quantize': True}      # 重度
]

results = {}
for i, config in enumerate(configs):
    model = create_compressed_model(base_model, config)
    metrics = evaluate_model(model, test_loader, device)
    results[f"Config_{i}"] = metrics
    print(f"Config {i}: Acc={metrics['accuracy']:.2f}%")
```

## 实验复现

### 运行完整实验套件

```bash
python experiments/run_experiments.py
```

这将:
1. 创建并评估基础模型
2. 测试多种压缩配置
3. 生成对比图表
4. 保存结果到 `experiments/results/`

### 查看结果

实验完成后，查看生成的文件：
- `experiment_results.json`: 详细数值结果
- `accuracy_comparison.png`: 准确率对比图
- `parameters_comparison.png`: 参数量对比图
- `inference_time_comparison.png`: 推理时间对比图
- `accuracy_vs_size.png`: 准确率-模型大小散点图

## API文档

### 模型类

#### TransformerEncoder
```python
class TransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size,          # 词汇表大小
        d_model=512,         # 模型维度
        num_heads=8,         # 注意力头数
        num_layers=6,        # 层数
        d_ff=2048,          # FFN维度
        max_len=5000,       # 最大序列长度
        dropout=0.1,        # Dropout率
        num_classes=2       # 分类类别数
    )
```

### 压缩函数

#### create_compressed_model
```python
def create_compressed_model(
    base_model,              # 基础模型
    compression_config       # 压缩配置字典
):
    """
    创建压缩模型
    
    compression_config示例:
    {
        'prune_heads': True,          # 是否剪枝注意力头
        'prune_ratio': 0.5,           # 剪枝比例
        'prune_layers': [4, 5],       # 要剪枝的层
        'quantize': True,             # 是否量化
        'quantize_bits': 8            # 量化位数
    }
    """
```

### 评估函数

#### evaluate_model
```python
def evaluate_model(model, data_loader, device):
    """
    评估模型性能
    
    返回:
    {
        'accuracy': float,      # 准确率(%)
        'loss': float,          # 平均损失
        'throughput': float,    # 吞吐量(样本/秒)
        'time': float          # 总时间(秒)
    }
    """
```

## 常见问题

### Q1: 如何选择合适的压缩配置？

**A:** 根据应用场景选择:
- **云端部署**: 中度压缩（75%参数减少）
- **边缘设备**: 重度压缩（85-90%参数减少）
- **离线处理**: 轻度压缩（30-50%参数减少）

### Q2: 压缩后准确率下降多少是可接受的？

**A:** 一般认为:
- < 1%: 优秀
- 1-2%: 良好
- 2-5%: 可接受
- > 5%: 需要调整压缩策略

### Q3: 知识蒸馏需要训练多久？

**A:** 通常是原始训练的50-70%时长。建议:
- 小模型: 5-8 epochs
- 中等模型: 8-12 epochs
- 大模型: 15-20 epochs

### Q4: 量化会影响训练吗？

**A:** 我们使用的是训练后量化(PTQ)，不影响训练。如果需要更高精度，可以使用量化感知训练(QAT)。

## 进阶使用

### 自定义压缩策略

```python
from src.compression.techniques import ModelPruner

class CustomPruner(ModelPruner):
    def custom_prune_strategy(self, model):
        # 实现自定义的剪枝逻辑
        pass
```

### 集成到生产环境

```python
# 加载压缩模型
model = torch.load('compressed_model.pt')
model.eval()

# 推理
with torch.no_grad():
    output = model(input_tensor)
    prediction = torch.argmax(output, dim=1)
```

## 引用

如果你在研究中使用了本项目，请引用:

```bibtex
@misc{transformer_compression_2024,
  title={Transformer Model Compression Toolkit},
  author={Transformer Compression Research Team},
  year={2024},
  url={https://github.com/1eleven112/ML1}
}
```

## 贡献

欢迎提交Issue和Pull Request!

## 许可证

MIT License

## 联系方式

研究方向：Transformer模型压缩
