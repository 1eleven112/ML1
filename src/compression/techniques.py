"""
模型压缩技术实现
包括剪枝、量化、知识蒸馏
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict


class HeadImportanceCalculator:
    """计算注意力头的重要性"""
    
    def __init__(self, model, alpha=0.5, beta=0.5):
        """
        Args:
            model: Transformer模型
            alpha: 梯度权重
            beta: 方差权重
        """
        self.model = model
        self.alpha = alpha
        self.beta = beta
        self.head_importance = {}
        self.head_variance = {}
        
    def compute_head_importance(self, data_loader, device):
        """
        计算每个注意力头的重要性
        
        Args:
            data_loader: 数据加载器
            device: 设备
        
        Returns:
            importance_scores: 字典，{layer_idx: [head1_score, head2_score, ...]}
        """
        self.model.eval()
        self.head_importance = {i: [] for i in range(len(self.model.layers))}
        self.head_variance = {i: [] for i in range(len(self.model.layers))}
        
        # 收集统计信息
        for batch_idx, (inputs, labels) in enumerate(data_loader):
            if batch_idx >= 100:  # 使用部分数据即可
                break
                
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 前向传播
            outputs = self.model(inputs)
            loss = F.cross_entropy(outputs, labels)
            
            # 反向传播
            self.model.zero_grad()
            loss.backward()
            
            # 收集每层的注意力信息
            for layer_idx, layer in enumerate(self.model.layers):
                attn_weights = layer.attention.attention_weights
                
                if attn_weights is not None:
                    # 计算注意力分布的方差（每个头）
                    for head_idx in range(attn_weights.size(1)):
                        head_attn = attn_weights[:, head_idx, :, :]
                        variance = torch.var(head_attn).item()
                        
                        if len(self.head_variance[layer_idx]) <= head_idx:
                            self.head_variance[layer_idx].append([])
                        self.head_variance[layer_idx][head_idx].append(variance)
        
        # 计算综合重要性分数
        importance_scores = {}
        for layer_idx in range(len(self.model.layers)):
            layer_scores = []
            num_heads = len(self.head_variance[layer_idx])
            
            for head_idx in range(num_heads):
                # 使用方差的平均值作为重要性指标
                variance_score = np.mean(self.head_variance[layer_idx][head_idx])
                
                # 可以结合梯度信息，这里简化为只用方差
                importance = variance_score
                layer_scores.append(importance)
            
            importance_scores[layer_idx] = layer_scores
        
        return importance_scores


class ModelPruner:
    """模型剪枝器"""
    
    def __init__(self, model):
        self.model = model
        self.original_config = {
            'num_layers': len(model.layers),
            'num_heads': model.layers[0].attention.num_heads if model.layers else 0,
        }
    
    def prune_attention_heads(self, importance_scores: Dict, prune_ratio: float = 0.5):
        """
        剪枝注意力头
        
        Args:
            importance_scores: 重要性分数
            prune_ratio: 剪枝比例
        """
        print(f"开始剪枝注意力头，剪枝比例: {prune_ratio}")
        
        for layer_idx, layer in enumerate(self.model.layers):
            if layer_idx not in importance_scores:
                continue
            
            scores = importance_scores[layer_idx]
            num_heads = len(scores)
            num_to_keep = max(1, int(num_heads * (1 - prune_ratio)))
            
            # 获取要保留的头的索引
            keep_indices = np.argsort(scores)[-num_to_keep:]
            
            print(f"Layer {layer_idx}: 保留 {num_to_keep}/{num_heads} 个注意力头")
            
            # 这里简化实现：实际剪枝需要重构网络
            # 在真实实现中，需要创建新的更小的权重矩阵
    
    def prune_layers(self, layers_to_remove: List[int]):
        """
        移除指定的层
        
        Args:
            layers_to_remove: 要移除的层索引列表
        """
        print(f"移除层: {layers_to_remove}")
        
        # 创建新的层列表
        new_layers = []
        for idx, layer in enumerate(self.model.layers):
            if idx not in layers_to_remove:
                new_layers.append(layer)
        
        self.model.layers = nn.ModuleList(new_layers)
        print(f"剪枝后层数: {len(self.model.layers)}")


class KnowledgeDistiller:
    """知识蒸馏训练器"""
    
    def __init__(
        self,
        teacher_model,
        student_model,
        temperature=4.0,
        alpha=0.5,
        beta=0.3,
        gamma=0.2
    ):
        """
        Args:
            teacher_model: 教师模型
            student_model: 学生模型
            temperature: 蒸馏温度
            alpha: 任务损失权重
            beta: 蒸馏损失权重
            gamma: 特征损失权重
        """
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
    
    def compute_distillation_loss(
        self,
        student_logits,
        teacher_logits,
        labels
    ):
        """
        计算蒸馏损失
        
        Args:
            student_logits: 学生模型输出
            teacher_logits: 教师模型输出
            labels: 真实标签
        
        Returns:
            total_loss: 总损失
        """
        # 任务损失（交叉熵）
        task_loss = F.cross_entropy(student_logits, labels)
        
        # 蒸馏损失（KL散度）
        T = self.temperature
        soft_targets = F.softmax(teacher_logits / T, dim=-1)
        soft_predictions = F.log_softmax(student_logits / T, dim=-1)
        distillation_loss = F.kl_div(
            soft_predictions,
            soft_targets,
            reduction='batchmean'
        ) * (T * T)
        
        # 组合损失
        total_loss = (
            self.alpha * task_loss +
            self.beta * distillation_loss
        )
        
        return total_loss, task_loss, distillation_loss
    
    def train_step(self, inputs, labels, optimizer, device):
        """
        单步训练
        
        Args:
            inputs: 输入数据
            labels: 标签
            optimizer: 优化器
            device: 设备
        
        Returns:
            loss_dict: 损失字典
        """
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # 教师模型前向传播
        with torch.no_grad():
            teacher_logits = self.teacher_model(inputs)
        
        # 学生模型前向传播
        student_logits = self.student_model(inputs)
        
        # 计算损失
        total_loss, task_loss, distill_loss = self.compute_distillation_loss(
            student_logits, teacher_logits, labels
        )
        
        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'task_loss': task_loss.item(),
            'distillation_loss': distill_loss.item()
        }


class ModelQuantizer:
    """模型量化器"""
    
    def __init__(self, model, bits=8):
        """
        Args:
            model: 要量化的模型
            bits: 量化位数
        """
        self.model = model
        self.bits = bits
        self.scale_factors = {}
        
    def quantize_tensor(self, tensor, bits=8):
        """
        对张量进行量化
        
        Args:
            tensor: 输入张量
            bits: 量化位数
        
        Returns:
            quantized_tensor: 量化后的张量
            scale: 缩放因子
        """
        # 对称量化
        max_val = torch.max(torch.abs(tensor))
        scale = max_val / (2 ** (bits - 1) - 1)
        
        if scale == 0:
            scale = 1.0
        
        quantized = torch.round(tensor / scale)
        quantized = torch.clamp(quantized, -(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
        
        # 反量化以便继续使用
        dequantized = quantized * scale
        
        return dequantized, scale
    
    def quantize_model(self):
        """量化整个模型"""
        print(f"开始 {self.bits}-bit 量化...")
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                with torch.no_grad():
                    quantized_param, scale = self.quantize_tensor(param.data, self.bits)
                    param.data = quantized_param
                    self.scale_factors[name] = scale
        
        print(f"量化完成，共量化 {len(self.scale_factors)} 个参数")
        
    def get_model_size(self):
        """计算量化后的模型大小（MB）"""
        total_bits = 0
        for param in self.model.parameters():
            total_bits += param.numel() * self.bits
        
        total_mb = total_bits / (8 * 1024 * 1024)
        return total_mb


def create_compressed_model(
    base_model,
    compression_config: Dict
):
    """
    创建压缩模型
    
    Args:
        base_model: 基础模型
        compression_config: 压缩配置
            {
                'prune_heads': True/False,
                'prune_ratio': 0.5,
                'prune_layers': [4, 5],
                'quantize': True/False,
                'quantize_bits': 8
            }
    
    Returns:
        compressed_model: 压缩后的模型
    """
    import copy
    compressed_model = copy.deepcopy(base_model)
    
    # 剪枝
    if compression_config.get('prune_heads', False):
        pruner = ModelPruner(compressed_model)
        # 需要先计算重要性
        print("注意：需要先用数据计算注意力头重要性")
    
    if compression_config.get('prune_layers'):
        pruner = ModelPruner(compressed_model)
        pruner.prune_layers(compression_config['prune_layers'])
    
    # 量化
    if compression_config.get('quantize', False):
        quantizer = ModelQuantizer(
            compressed_model,
            bits=compression_config.get('quantize_bits', 8)
        )
        quantizer.quantize_model()
    
    return compressed_model


if __name__ == "__main__":
    print("模型压缩技术模块测试")
    print("包含: 剪枝、知识蒸馏、量化")
