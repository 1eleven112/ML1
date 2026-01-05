"""
训练基础Transformer模型
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import sys

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.transformer import create_base_model
from utils.helpers import (
    create_dummy_dataset,
    SimpleTokenizer,
    TextDataset,
    evaluate_model,
    print_model_summary,
    save_checkpoint,
    AverageMeter
)


def train_epoch(model, data_loader, criterion, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    
    losses = AverageMeter()
    accuracies = AverageMeter()
    
    for batch_idx, (inputs, labels) in enumerate(data_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == labels).float().mean().item() * 100
        
        losses.update(loss.item(), inputs.size(0))
        accuracies.update(accuracy, inputs.size(0))
        
        if batch_idx % 10 == 0:
            print(f'Epoch: [{epoch}][{batch_idx}/{len(data_loader)}]\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                  f'Acc {accuracies.val:.2f}% ({accuracies.avg:.2f}%)')
    
    return losses.avg, accuracies.avg


def main(args):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建数据集
    print("创建数据集...")
    train_texts, train_labels = create_dummy_dataset(
        num_samples=args.num_samples,
        num_classes=args.num_classes
    )
    test_texts, test_labels = create_dummy_dataset(
        num_samples=args.num_samples // 5,
        num_classes=args.num_classes
    )
    
    # 构建词汇表
    tokenizer = SimpleTokenizer(vocab_size=args.vocab_size)
    tokenizer.build_vocab(train_texts)
    vocab_size = len(tokenizer.word2idx)
    print(f"词汇表大小: {vocab_size}")
    
    # 创建数据加载器
    train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_len=args.max_len)
    test_dataset = TextDataset(test_texts, test_labels, tokenizer, max_len=args.max_len)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # 创建模型
    print("\n创建模型...")
    model = create_base_model(vocab_size, num_classes=args.num_classes)
    
    # 加载预训练模型（如果指定）
    if args.pretrained_model:
        if os.path.exists(args.pretrained_model):
            print(f"加载预训练模型: {args.pretrained_model}")
            checkpoint = torch.load(args.pretrained_model, map_location='cpu')
            
            # 处理不同的保存格式
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"从checkpoint加载模型 (epoch {checkpoint.get('epoch', 'N/A')})")
            else:
                model.load_state_dict(checkpoint)
                print("从state_dict加载模型")
        else:
            print(f"警告: 预训练模型文件不存在: {args.pretrained_model}")
            print("将使用随机初始化的模型")
    
    model = model.to(device)
    print_model_summary(model, "基础Transformer模型")
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # 训练循环
    print("\n开始训练...")
    best_acc = 0.0
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        print("-" * 60)
        
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # 评估
        test_metrics = evaluate_model(model, test_loader, device)
        test_acc = test_metrics['accuracy']
        test_loss = test_metrics['loss']
        
        print(f"\nEpoch {epoch + 1} 结果:")
        print(f"训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"测试 - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            save_path = os.path.join(args.save_dir, 'best_base_model.pt')
            save_checkpoint(model, optimizer, epoch, test_loss, save_path)
            print(f"保存最佳模型，准确率: {best_acc:.2f}%")
        
        # 更新学习率
        scheduler.step()
    
    # 最终评估
    print("\n" + "="*60)
    print("训练完成!")
    print(f"最佳测试准确率: {best_acc:.2f}%")
    print("="*60)
    
    # 保存最终模型
    final_path = os.path.join(args.save_dir, 'final_base_model.pt')
    torch.save(model.state_dict(), final_path)
    print(f"最终模型已保存到: {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='训练基础Transformer模型')
    
    # 数据参数
    parser.add_argument('--num_samples', type=int, default=5000,
                        help='训练样本数量')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='分类类别数')
    parser.add_argument('--vocab_size', type=int, default=10000,
                        help='词汇表大小')
    parser.add_argument('--max_len', type=int, default=128,
                        help='最大序列长度')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='学习率')
    
    # 其他参数
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='模型保存目录')
    parser.add_argument('--pretrained_model', type=str, default=None,
                        help='预训练模型路径（可选）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    args = parser.parse_args()
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # 开始训练
    main(args)
