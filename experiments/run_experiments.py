"""
运行完整的实验套件
对比不同压缩方法的效果
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import sys
import json
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.transformer import create_base_model
from src.compression.techniques import create_compressed_model
from src.utils.helpers import (
    create_dummy_dataset,
    SimpleTokenizer,
    TextDataset,
    evaluate_model,
    measure_inference_time,
    count_parameters,
    get_model_size
)


class ExperimentRunner:
    """实验运行器"""
    
    def __init__(self, vocab_size, num_classes, device):
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.device = device
        self.results = {}
    
    def run_experiment(self, model, model_name, test_loader):
        """运行单个实验"""
        print(f"\n{'='*60}")
        print(f"评估: {model_name}")
        print(f"{'='*60}")
        
        model = model.to(self.device)
        
        # 评估准确率
        metrics = evaluate_model(model, test_loader, self.device)
        
        # 测量推理时间
        avg_time, std_time = measure_inference_time(
            model, test_loader, self.device, num_iterations=50
        )
        
        # 模型统计
        num_params = count_parameters(model)
        model_size_fp32 = get_model_size(model, bits=32)
        model_size_int8 = get_model_size(model, bits=8)
        
        results = {
            'accuracy': metrics['accuracy'],
            'loss': metrics['loss'],
            'num_parameters': num_params,
            'model_size_mb_fp32': model_size_fp32,
            'model_size_mb_int8': model_size_int8,
            'inference_time_ms': avg_time,
            'inference_std_ms': std_time,
            'throughput': metrics['throughput']
        }
        
        # 打印结果
        print(f"\n参数量: {num_params:,}")
        print(f"模型大小 (FP32): {model_size_fp32:.2f} MB")
        print(f"模型大小 (INT8): {model_size_int8:.2f} MB")
        print(f"准确率: {metrics['accuracy']:.2f}%")
        print(f"推理时间: {avg_time:.2f} ± {std_time:.2f} ms")
        print(f"吞吐量: {metrics['throughput']:.1f} 样本/秒")
        
        self.results[model_name] = results
        return results
    
    def compare_models(self):
        """对比所有模型的结果"""
        if not self.results:
            print("没有可对比的结果")
            return
        
        print(f"\n{'='*60}")
        print("模型对比")
        print(f"{'='*60}")
        
        # 创建对比表格
        print(f"\n{'模型':<25} {'参数量':<12} {'准确率':<10} {'推理时间':<12} {'模型大小(MB)':<15}")
        print("-" * 90)
        
        baseline_params = None
        baseline_acc = None
        
        for model_name, results in self.results.items():
            if baseline_params is None:
                baseline_params = results['num_parameters']
                baseline_acc = results['accuracy']
            
            params_str = f"{results['num_parameters']/1e6:.1f}M"
            acc_str = f"{results['accuracy']:.2f}%"
            time_str = f"{results['inference_time_ms']:.2f}ms"
            size_str = f"{results['model_size_mb_int8']:.2f}"
            
            print(f"{model_name:<25} {params_str:<12} {acc_str:<10} {time_str:<12} {size_str:<15}")
        
        print("\n" + "="*60)
    
    def save_results(self, filepath='experiments/results/experiment_results.json'):
        """保存实验结果"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"\n实验结果已保存到: {filepath}")
    
    def plot_results(self, save_dir='experiments/results'):
        """绘制结果图表"""
        if not self.results:
            print("没有可绘制的结果")
            return
        
        os.makedirs(save_dir, exist_ok=True)
        
        model_names = list(self.results.keys())
        accuracies = [self.results[name]['accuracy'] for name in model_names]
        params = [self.results[name]['num_parameters']/1e6 for name in model_names]
        times = [self.results[name]['inference_time_ms'] for name in model_names]
        sizes = [self.results[name]['model_size_mb_int8'] for name in model_names]
        
        # 图1: 准确率对比
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(model_names)), accuracies, color='skyblue')
        plt.xlabel('模型', fontproperties='SimHei')
        plt.ylabel('准确率 (%)', fontproperties='SimHei')
        plt.title('不同模型的准确率对比', fontproperties='SimHei')
        plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'accuracy_comparison.png'), dpi=300)
        plt.close()
        
        # 图2: 参数量对比
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(model_names)), params, color='lightcoral')
        plt.xlabel('模型', fontproperties='SimHei')
        plt.ylabel('参数量 (百万)', fontproperties='SimHei')
        plt.title('不同模型的参数量对比', fontproperties='SimHei')
        plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'parameters_comparison.png'), dpi=300)
        plt.close()
        
        # 图3: 推理时间对比
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(model_names)), times, color='lightgreen')
        plt.xlabel('模型', fontproperties='SimHei')
        plt.ylabel('推理时间 (ms)', fontproperties='SimHei')
        plt.title('不同模型的推理时间对比', fontproperties='SimHei')
        plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'inference_time_comparison.png'), dpi=300)
        plt.close()
        
        # 图4: 准确率 vs 模型大小
        plt.figure(figsize=(10, 6))
        plt.scatter(sizes, accuracies, s=100, alpha=0.6, c=range(len(model_names)), cmap='viridis')
        for i, name in enumerate(model_names):
            plt.annotate(name, (sizes[i], accuracies[i]), fontsize=8, ha='right')
        plt.xlabel('模型大小 (MB)', fontproperties='SimHei')
        plt.ylabel('准确率 (%)', fontproperties='SimHei')
        plt.title('准确率 vs 模型大小', fontproperties='SimHei')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'accuracy_vs_size.png'), dpi=300)
        plt.close()
        
        print(f"图表已保存到: {save_dir}")


def main():
    print("="*60)
    print("Transformer模型压缩实验套件")
    print("="*60)
    
    # 设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 创建数据集
    print("\n创建测试数据集...")
    test_texts, test_labels = create_dummy_dataset(num_samples=1000, num_classes=2)
    
    tokenizer = SimpleTokenizer(vocab_size=10000)
    tokenizer.build_vocab(test_texts)
    vocab_size = len(tokenizer.word2idx)
    
    test_dataset = TextDataset(test_texts, test_labels, tokenizer, max_len=128)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 创建实验运行器
    runner = ExperimentRunner(vocab_size, num_classes=2, device=device)
    
    # 实验1: 基础模型
    print("\n实验1: 基础模型")
    base_model = create_base_model(vocab_size, num_classes=2)
    runner.run_experiment(base_model, "基础模型", test_loader)
    
    # 实验2: 轻度压缩（层剪枝）
    print("\n实验2: 轻度压缩")
    light_config = {
        'prune_layers': [5],  # 只剪枝一层
        'quantize': False
    }
    light_model = create_compressed_model(base_model, light_config)
    runner.run_experiment(light_model, "轻度压缩", test_loader)
    
    # 实验3: 中度压缩（层剪枝+量化）
    print("\n实验3: 中度压缩")
    medium_config = {
        'prune_layers': [4, 5],  # 剪枝两层
        'quantize': True,
        'quantize_bits': 8
    }
    medium_model = create_compressed_model(base_model, medium_config)
    runner.run_experiment(medium_model, "中度压缩", test_loader)
    
    # 实验4: 重度压缩
    print("\n实验4: 重度压缩")
    heavy_config = {
        'prune_layers': [3, 4, 5],  # 剪枝三层
        'quantize': True,
        'quantize_bits': 8
    }
    heavy_model = create_compressed_model(base_model, heavy_config)
    runner.run_experiment(heavy_model, "重度压缩", test_loader)
    
    # 对比和保存结果
    runner.compare_models()
    runner.save_results()
    
    # 绘制图表（注意：在没有中文字体的环境中可能显示不正常）
    try:
        runner.plot_results()
    except Exception as e:
        print(f"绘图失败: {e}")
        print("这可能是因为系统没有中文字体，但实验结果已正确保存")
    
    print("\n" + "="*60)
    print("实验完成!")
    print("="*60)


if __name__ == "__main__":
    main()
