"""
训练压缩模型
支持剪枝、量化、知识蒸馏
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.transformer import create_base_model
from compression.techniques import (
    HeadImportanceCalculator,
    ModelPruner,
    KnowledgeDistiller,
    ModelQuantizer,
    create_compressed_model
)
from utils.helpers import (
    create_dummy_dataset,
    SimpleTokenizer,
    TextDataset,
    evaluate_model,
    print_model_summary,
    save_checkpoint,
    AverageMeter
)


def train_with_distillation(
    teacher_model,
    student_model,
    train_loader,
    test_loader,
    device,
    args
):
    """使用知识蒸馏训练学生模型"""
    
    print("\n" + "="*60)
    print("开始知识蒸馏训练")
    print("="*60)
    
    # 创建蒸馏器
    distiller = KnowledgeDistiller(
        teacher_model=teacher_model,
        student_model=student_model,
        temperature=args.temperature,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma
    )
    
    optimizer = optim.Adam(student_model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    
    best_acc = 0.0
    
    for epoch in range(args.distill_epochs):
        print(f"\nDistillation Epoch {epoch + 1}/{args.distill_epochs}")
        print("-" * 60)
        
        student_model.train()
        
        losses = AverageMeter()
        task_losses = AverageMeter()
        distill_losses = AverageMeter()
        accuracies = AverageMeter()
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            # 训练步骤
            loss_dict = distiller.train_step(inputs, labels, optimizer, device)
            
            # 计算准确率
            with torch.no_grad():
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = student_model(inputs)
                _, predicted = torch.max(outputs, 1)
                accuracy = (predicted == labels).float().mean().item() * 100
            
            # 更新统计
            losses.update(loss_dict['total_loss'], inputs.size(0))
            task_losses.update(loss_dict['task_loss'], inputs.size(0))
            distill_losses.update(loss_dict['distillation_loss'], inputs.size(0))
            accuracies.update(accuracy, inputs.size(0))
            
            if batch_idx % 10 == 0:
                print(f'[{batch_idx}/{len(train_loader)}]\t'
                      f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                      f'Task {task_losses.val:.4f} Distill {distill_losses.val:.4f}\t'
                      f'Acc {accuracies.val:.2f}% ({accuracies.avg:.2f}%)')
        
        # 评估
        test_metrics = evaluate_model(student_model, test_loader, device)
        test_acc = test_metrics['accuracy']
        test_loss = test_metrics['loss']
        
        print(f"\nEpoch {epoch + 1} 结果:")
        print(f"训练 - Total Loss: {losses.avg:.4f}, Acc: {accuracies.avg:.2f}%")
        print(f"测试 - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            save_path = os.path.join(args.save_dir, 'best_compressed_model.pt')
            save_checkpoint(student_model, optimizer, epoch, test_loss, save_path)
            print(f"保存最佳压缩模型，准确率: {best_acc:.2f}%")
        
        scheduler.step()
    
    return best_acc


def main(args):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建数据集
    print("\n创建数据集...")
    train_texts, train_labels = create_dummy_dataset(
        num_samples=args.num_samples,
        num_classes=args.num_classes
    )
    test_texts, test_labels = create_dummy_dataset(
        num_samples=args.num_samples // 5,
        num_classes=args.num_classes
    )
    
    tokenizer = SimpleTokenizer(vocab_size=args.vocab_size)
    tokenizer.build_vocab(train_texts)
    vocab_size = len(tokenizer.word2idx)
    
    train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_len=args.max_len)
    test_dataset = TextDataset(test_texts, test_labels, tokenizer, max_len=args.max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 加载或创建教师模型（基础模型）
    print("\n加载教师模型...")
    teacher_model = create_base_model(vocab_size, num_classes=args.num_classes)
    
    if args.teacher_model_path and os.path.exists(args.teacher_model_path):
        teacher_model.load_state_dict(torch.load(args.teacher_model_path, map_location=device))
        print(f"从 {args.teacher_model_path} 加载教师模型")
    else:
        print("警告: 未找到预训练的教师模型，使用随机初始化的模型")
    
    teacher_model = teacher_model.to(device)
    teacher_model.eval()
    
    # 评估教师模型
    teacher_metrics = evaluate_model(teacher_model, test_loader, device)
    print(f"\n教师模型性能: 准确率 {teacher_metrics['accuracy']:.2f}%")
    print_model_summary(teacher_model, "教师模型")
    
    # 创建学生模型（压缩模型）
    print("\n创建压缩的学生模型...")
    
    compression_config = {
        'prune_layers': args.prune_layers,
        'quantize': args.quantize,
        'quantize_bits': args.quantize_bits
    }
    
    student_model = create_compressed_model(teacher_model, compression_config)
    student_model = student_model.to(device)
    print_model_summary(student_model, "学生模型（压缩）")
    
    # 计算压缩比
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    student_params = sum(p.numel() for p in student_model.parameters())
    compression_ratio = (1 - student_params / teacher_params) * 100
    print(f"\n压缩比: {compression_ratio:.1f}% (参数减少)")
    print(f"教师模型参数: {teacher_params:,}")
    print(f"学生模型参数: {student_params:,}")
    
    # 评估压缩前的学生模型
    print("\n评估压缩后、蒸馏前的模型...")
    before_metrics = evaluate_model(student_model, test_loader, device)
    print(f"蒸馏前准确率: {before_metrics['accuracy']:.2f}%")
    
    # 知识蒸馏训练
    if args.use_distillation:
        best_acc = train_with_distillation(
            teacher_model,
            student_model,
            train_loader,
            test_loader,
            device,
            args
        )
    else:
        print("\n跳过知识蒸馏，直接保存压缩模型")
        best_acc = before_metrics['accuracy']
    
    # 量化（如果需要）
    if args.quantize:
        print("\n" + "="*60)
        print("应用量化...")
        quantizer = ModelQuantizer(student_model, bits=args.quantize_bits)
        quantizer.quantize_model()
        
        # 评估量化后的模型
        quant_metrics = evaluate_model(student_model, test_loader, device)
        print(f"量化后准确率: {quant_metrics['accuracy']:.2f}%")
    
    # 最终评估和对比
    print("\n" + "="*60)
    print("最终结果对比")
    print("="*60)
    
    final_metrics = evaluate_model(student_model, test_loader, device)
    
    print(f"\n教师模型:")
    print(f"  - 参数量: {teacher_params:,}")
    print(f"  - 准确率: {teacher_metrics['accuracy']:.2f}%")
    
    print(f"\n学生模型（压缩）:")
    print(f"  - 参数量: {student_params:,}")
    print(f"  - 压缩比: {compression_ratio:.1f}%")
    print(f"  - 准确率: {final_metrics['accuracy']:.2f}%")
    print(f"  - 准确率下降: {teacher_metrics['accuracy'] - final_metrics['accuracy']:.2f}%")
    
    # 保存最终模型
    final_path = os.path.join(args.save_dir, 'final_compressed_model.pt')
    torch.save(student_model.state_dict(), final_path)
    print(f"\n最终压缩模型已保存到: {final_path}")
    
    print("\n" + "="*60)
    print("压缩完成!")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='训练压缩的Transformer模型')
    
    # 数据参数
    parser.add_argument('--num_samples', type=int, default=5000)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--vocab_size', type=int, default=10000)
    parser.add_argument('--max_len', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=32)
    
    # 模型参数
    parser.add_argument('--teacher_model_path', type=str, default='checkpoints/final_base_model.pt',
                        help='教师模型路径')
    
    # 压缩参数
    parser.add_argument('--method', type=str, default='hybrid',
                        choices=['pruning', 'distillation', 'quantization', 'hybrid'],
                        help='压缩方法')
    parser.add_argument('--prune_layers', type=int, nargs='+', default=[4, 5],
                        help='要剪枝的层索引')
    parser.add_argument('--quantize', action='store_true', default=True,
                        help='是否量化')
    parser.add_argument('--quantize_bits', type=int, default=8,
                        help='量化位数')
    
    # 蒸馏参数
    parser.add_argument('--use_distillation', action='store_true', default=True,
                        help='是否使用知识蒸馏')
    parser.add_argument('--distill_epochs', type=int, default=8,
                        help='蒸馏训练轮数')
    parser.add_argument('--temperature', type=float, default=4.0,
                        help='蒸馏温度')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='任务损失权重')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='蒸馏损失权重')
    parser.add_argument('--gamma', type=float, default=0.0,
                        help='特征损失权重')
    
    # 训练参数
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    main(args)
