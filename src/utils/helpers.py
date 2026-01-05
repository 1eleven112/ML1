"""
工具函数
数据加载、评估等
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple
import time


class TextDataset(Dataset):
    """简单的文本分类数据集"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 简单的tokenization（实际应该使用更好的tokenizer）
        tokens = self.tokenizer(text, self.max_len)
        
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(label, dtype=torch.long)


class SimpleTokenizer:
    """简单的分词器（演示用）"""
    
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.word2idx = {}
        self.idx2word = {}
        
    def build_vocab(self, texts: List[str]):
        """构建词汇表"""
        word_freq = {}
        for text in texts:
            words = text.lower().split()
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # 选择最常见的词
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # 特殊token
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        
        for idx, (word, _) in enumerate(sorted_words[:self.vocab_size - 2], start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
    
    def __call__(self, text: str, max_len: int = 128):
        """将文本转换为token ids"""
        words = text.lower().split()
        tokens = [self.word2idx.get(word, 1) for word in words]  # 1 is <UNK>
        
        # 截断或填充
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        else:
            tokens = tokens + [0] * (max_len - len(tokens))  # 0 is <PAD>
        
        return tokens


def create_dummy_dataset(num_samples=1000, num_classes=2):
    """
    创建虚拟数据集用于演示
    
    Args:
        num_samples: 样本数量
        num_classes: 类别数量
    
    Returns:
        texts, labels: 文本和标签列表
    """
    texts = []
    labels = []
    
    # 简单的情感分析数据
    positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'best']
    negative_words = ['bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'disappointing', 'poor']
    neutral_words = ['movie', 'film', 'show', 'actor', 'story', 'plot', 'scene', 'character']
    
    for i in range(num_samples):
        label = i % num_classes
        
        # 生成文本
        if label == 0:  # 负面
            words = np.random.choice(negative_words, size=np.random.randint(10, 20))
        else:  # 正面
            words = np.random.choice(positive_words, size=np.random.randint(10, 20))
        
        # 添加一些中性词
        words = list(words) + list(np.random.choice(neutral_words, size=5))
        np.random.shuffle(words)
        
        text = ' '.join(words)
        texts.append(text)
        labels.append(label)
    
    return texts, labels


def evaluate_model(model, data_loader, device):
    """
    评估模型性能
    
    Args:
        model: 模型
        data_loader: 数据加载器
        device: 设备
    
    Returns:
        metrics: 评估指标字典
    """
    model.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    start_time = time.time()
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    elapsed_time = time.time() - start_time
    
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(data_loader)
    throughput = total / elapsed_time
    
    metrics = {
        'accuracy': accuracy,
        'loss': avg_loss,
        'correct': correct,
        'total': total,
        'throughput': throughput,
        'time': elapsed_time
    }
    
    return metrics


def measure_inference_time(model, data_loader, device, num_iterations=100):
    """
    测量推理时间
    
    Args:
        model: 模型
        data_loader: 数据加载器
        device: 设备
        num_iterations: 迭代次数
    
    Returns:
        avg_time: 平均推理时间（ms）
    """
    model.eval()
    
    times = []
    
    with torch.no_grad():
        for i, (inputs, _) in enumerate(data_loader):
            if i >= num_iterations:
                break
            
            inputs = inputs.to(device)
            
            # 预热
            if i < 10:
                _ = model(inputs)
                continue
            
            # 计时
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.time()
            _ = model(inputs)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            elapsed = (time.time() - start_time) * 1000  # 转换为ms
            times.append(elapsed)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    return avg_time, std_time


def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size(model, bits=32):
    """
    计算模型大小
    
    Args:
        model: 模型
        bits: 每个参数的位数
    
    Returns:
        size_mb: 模型大小（MB）
    """
    num_params = count_parameters(model)
    size_mb = (num_params * bits) / (8 * 1024 * 1024)
    return size_mb


def print_model_summary(model, model_name="Model"):
    """打印模型摘要"""
    print(f"\n{'='*60}")
    print(f"{model_name} 摘要")
    print(f"{'='*60}")
    print(f"参数量: {count_parameters(model):,}")
    print(f"模型大小 (FP32): {get_model_size(model, 32):.2f} MB")
    print(f"模型大小 (FP16): {get_model_size(model, 16):.2f} MB")
    print(f"模型大小 (INT8): {get_model_size(model, 8):.2f} MB")
    print(f"{'='*60}\n")


class AverageMeter:
    """计算和存储平均值"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """保存模型检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    print(f"检查点已保存到: {filepath}")


def load_checkpoint(model, optimizer, filepath, device):
    """加载模型检查点"""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"从 {filepath} 加载检查点 (epoch {epoch})")
    return epoch, loss


if __name__ == "__main__":
    # 测试工具函数
    print("创建虚拟数据集...")
    texts, labels = create_dummy_dataset(num_samples=100)
    print(f"数据集大小: {len(texts)}")
    print(f"示例文本: {texts[0]}")
    print(f"示例标签: {labels[0]}")
