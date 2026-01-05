"""
使用预训练模型的示例脚本
演示如何加载和使用预训练的Transformer模型
"""

import torch
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.transformer import load_pretrained_model, create_base_model
from src.utils.helpers import create_dummy_dataset, SimpleTokenizer, TextDataset, evaluate_model
from torch.utils.data import DataLoader


def example1_load_and_inference():
    """示例1: 加载预训练模型并进行推理"""
    print("="*60)
    print("示例1: 加载预训练模型并进行推理")
    print("="*60)
    
    vocab_size = 10000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 假设我们有一个预训练模型
    pretrained_path = 'checkpoints/best_base_model.pt'
    
    if os.path.exists(pretrained_path):
        # 加载预训练模型
        model = load_pretrained_model(
            model_path=pretrained_path,
            vocab_size=vocab_size,
            num_classes=2,
            device=device
        )
        
        # 推理模式
        model.eval()
        
        # 创建示例输入
        sample_input = torch.randint(0, vocab_size, (1, 128)).to(device)
        
        with torch.no_grad():
            output = model(sample_input)
            prediction = torch.argmax(output, dim=1)
            probabilities = torch.softmax(output, dim=1)
        
        print(f"\n推理结果:")
        print(f"  预测类别: {prediction.item()}")
        print(f"  概率分布: {probabilities[0].cpu().numpy()}")
    else:
        print(f"预训练模型不存在: {pretrained_path}")
        print("请先训练一个基础模型: python src/training/train_base.py")


def example2_finetune_pretrained():
    """示例2: 在预训练模型基础上进行微调"""
    print("\n" + "="*60)
    print("示例2: 微调预训练模型")
    print("="*60)
    
    vocab_size = 10000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    pretrained_path = 'checkpoints/best_base_model.pt'
    
    if os.path.exists(pretrained_path):
        # 加载预训练模型
        model = load_pretrained_model(
            model_path=pretrained_path,
            vocab_size=vocab_size,
            num_classes=2,
            device=device
        )
        
        # 准备少量数据进行微调
        print("\n准备微调数据...")
        texts, labels = create_dummy_dataset(num_samples=100, num_classes=2)
        tokenizer = SimpleTokenizer(vocab_size=vocab_size)
        tokenizer.build_vocab(texts)
        
        dataset = TextDataset(texts, labels, tokenizer, max_len=128)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        # 设置优化器（使用较小的学习率进行微调）
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        criterion = torch.nn.CrossEntropyLoss()
        
        # 微调几个epoch
        print("\n开始微调...")
        model.train()
        for epoch in range(2):
            total_loss = 0
            for batch_idx, (inputs, labels) in enumerate(loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch+1}/2, Loss: {avg_loss:.4f}")
        
        print("\n微调完成!")
        
        # 保存微调后的模型
        finetuned_path = 'checkpoints/finetuned_model.pt'
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(model.state_dict(), finetuned_path)
        print(f"微调后的模型已保存到: {finetuned_path}")
    else:
        print(f"预训练模型不存在: {pretrained_path}")
        print("请先训练一个基础模型: python src/training/train_base.py")


def example3_compare_random_vs_pretrained():
    """示例3: 对比随机初始化和预训练模型的性能"""
    print("\n" + "="*60)
    print("示例3: 对比随机初始化 vs 预训练模型")
    print("="*60)
    
    vocab_size = 10000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 准备测试数据
    print("\n准备测试数据...")
    texts, labels = create_dummy_dataset(num_samples=200, num_classes=2)
    tokenizer = SimpleTokenizer(vocab_size=vocab_size)
    tokenizer.build_vocab(texts)
    
    dataset = TextDataset(texts, labels, tokenizer, max_len=128)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # 1. 随机初始化模型
    print("\n评估随机初始化模型...")
    random_model = create_base_model(vocab_size, num_classes=2).to(device)
    random_metrics = evaluate_model(random_model, loader, device)
    
    # 2. 预训练模型（如果存在）
    pretrained_path = 'checkpoints/best_base_model.pt'
    if os.path.exists(pretrained_path):
        print("\n评估预训练模型...")
        pretrained_model = load_pretrained_model(
            model_path=pretrained_path,
            vocab_size=vocab_size,
            num_classes=2,
            device=device
        )
        pretrained_metrics = evaluate_model(pretrained_model, loader, device)
        
        # 对比结果
        print("\n" + "="*60)
        print("性能对比:")
        print("="*60)
        print(f"{'模型':<20} {'准确率':<15} {'吞吐量 (样本/秒)':<20}")
        print("-"*60)
        print(f"{'随机初始化':<20} {random_metrics['accuracy']:<15.2f} {random_metrics['throughput']:<20.1f}")
        print(f"{'预训练模型':<20} {pretrained_metrics['accuracy']:<15.2f} {pretrained_metrics['throughput']:<20.1f}")
        print("-"*60)
        
        improvement = pretrained_metrics['accuracy'] - random_metrics['accuracy']
        print(f"\n预训练模型准确率提升: {improvement:.2f}%")
    else:
        print("\n预训练模型不存在，跳过对比")
        print(f"随机初始化模型准确率: {random_metrics['accuracy']:.2f}%")


if __name__ == "__main__":
    print("Transformer预训练模型使用示例")
    print("="*60)
    
    # 运行所有示例
    example1_load_and_inference()
    example2_finetune_pretrained()
    example3_compare_random_vs_pretrained()
    
    print("\n" + "="*60)
    print("所有示例完成!")
    print("="*60)
    print("\n提示:")
    print("1. 如果预训练模型不存在，请先运行: python src/training/train_base.py")
    print("2. 可以单独运行某个示例函数来测试特定功能")
    print("3. 更多详情请查看 docs/usage_guide.md")
