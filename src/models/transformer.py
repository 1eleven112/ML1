"""
基础Transformer模型实现
包含Multi-Head Attention、Position Encoding等核心组件
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """位置编码层"""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 线性变换层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None  # 用于可视化
        
    def split_heads(self, x):
        """将输入分割成多个头"""
        batch_size = x.size(0)
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(1, 2)  # [batch_size, num_heads, seq_len, d_k]
    
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: [batch_size, seq_len, d_model]
            key: [batch_size, seq_len, d_model]
            value: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len, seq_len]
        """
        batch_size = query.size(0)
        
        # 线性变换
        Q = self.split_heads(self.W_q(query))  # [batch_size, num_heads, seq_len, d_k]
        K = self.split_heads(self.W_k(key))
        V = self.split_heads(self.W_v(value))
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 应用mask（如果有）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax获得注意力权重
        attention = F.softmax(scores, dim=-1)
        self.attention_weights = attention.detach()  # 保存用于分析
        attention = self.dropout(attention)
        
        # 应用注意力权重
        context = torch.matmul(attention, V)  # [batch_size, num_heads, seq_len, d_k]
        
        # 合并多头
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, -1, self.d_model)
        
        # 最终线性变换
        output = self.W_o(context)
        
        return output


class FeedForward(nn.Module):
    """前馈神经网络"""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        """
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer编码器块"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        
        # 多头注意力
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # 前馈网络
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len, seq_len]
        """
        # 多头注意力 + 残差连接 + LayerNorm
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络 + 残差连接 + LayerNorm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class TransformerEncoder(nn.Module):
    """Transformer编码器"""
    
    def __init__(
        self,
        vocab_size,
        d_model=512,
        num_heads=8,
        num_layers=6,
        d_ff=2048,
        max_len=5000,
        dropout=0.1,
        num_classes=2
    ):
        super(TransformerEncoder, self).__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Embedding层
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer块
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 分类头
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)
        
        # 初始化参数
        self._init_parameters()
        
    def _init_parameters(self):
        """初始化模型参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, seq_len] - 输入token ids
            mask: [batch_size, seq_len, seq_len] - 注意力mask
        
        Returns:
            output: [batch_size, num_classes] - 分类logits
        """
        # Embedding + 位置编码
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        # 通过所有Transformer块
        for layer in self.layers:
            x = layer(x, mask)
        
        # 全局平均池化
        x = x.mean(dim=1)  # [batch_size, d_model]
        
        # 分类
        x = self.dropout(x)
        logits = self.fc(x)  # [batch_size, num_classes]
        
        return logits
    
    def get_attention_weights(self):
        """获取所有层的注意力权重（用于可视化）"""
        weights = []
        for layer in self.layers:
            if layer.attention.attention_weights is not None:
                weights.append(layer.attention.attention_weights.cpu())
        return weights
    
    def count_parameters(self):
        """计算模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_base_model(vocab_size, num_classes=2):
    """创建基础Transformer模型"""
    model = TransformerEncoder(
        vocab_size=vocab_size,
        d_model=512,
        num_heads=8,
        num_layers=6,
        d_ff=2048,
        dropout=0.1,
        num_classes=num_classes
    )
    return model


def load_pretrained_model(model_path, vocab_size, num_classes=2, device='cpu'):
    """
    加载预训练的Transformer模型
    
    Args:
        model_path: 预训练模型文件路径
        vocab_size: 词汇表大小
        num_classes: 分类类别数
        device: 设备 ('cpu' 或 'cuda')
    
    Returns:
        model: 加载了预训练权重的模型
    
    示例:
        # 加载预训练模型
        model = load_pretrained_model('checkpoints/best_model.pt', vocab_size=10000)
        
        # 继续训练或推理
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
    """
    import os
    
    # 创建模型结构
    model = create_base_model(vocab_size, num_classes)
    
    # 检查文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"预训练模型文件不存在: {model_path}")
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    
    # 处理不同的保存格式
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"从checkpoint加载模型 (epoch {checkpoint.get('epoch', 'N/A')})")
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # 假设字典本身就是state_dict
            model.load_state_dict(checkpoint)
    else:
        # 直接是state_dict
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    print(f"成功加载预训练模型: {model_path}")
    
    return model


if __name__ == "__main__":
    # 测试代码
    vocab_size = 10000
    batch_size = 32
    seq_len = 128
    
    model = create_base_model(vocab_size, num_classes=2)
    print(f"模型参数量: {model.count_parameters():,}")
    
    # 创建随机输入
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # 前向传播
    output = model(x)
    print(f"输出形状: {output.shape}")
    print(f"预期形状: [{batch_size}, 2]")
