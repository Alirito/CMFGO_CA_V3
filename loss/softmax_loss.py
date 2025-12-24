import torch
import torch.nn as nn
from torch.nn import functional as F
class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    # 定义初始化方法，接收类别数、平滑系数和GPU使用标志
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        # 创建log softmax层，在维度1上计算
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            输入: 形状为(批量大小, 类别数)的预测矩阵(softmax前)
            targets: ground truth labels with shape (num_classes)
            目标: 具有形状(num_classes)的真实标签
        """
        # 对输入计算log softmax，得到对数概率
        log_probs = self.logsoftmax(inputs)
        # 创建one-hot标签
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        
        # y = (1-ε)*y + ε/K，将硬标签变为软标签，防止过拟合
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        # 计算交叉熵损失：-Σ(y * log(p))，先按第0维（batch维度）求平均，然后求和
        loss = (- targets * log_probs).mean(0).sum()
        return loss

# 定义另一个标签平滑交叉熵损失类
class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        # 断言平滑系数必须小于1.0
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        # 在最后一个维度计算log softmax
        logprobs = F.log_softmax(x, dim=-1)
        # 使用gather方法获取目标类别对应的log概率，target.unsqueeze(1)将标签从[batch_size]变为[batch_size, 1]
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        # 将[batch_size, 1]压缩为[batch_size]
        nll_loss = nll_loss.squeeze(1)
        # 计算所有类别的平均负log概率
        smooth_loss = -logprobs.mean(dim=-1)
        # 将真实类别损失和平滑损失按权重组合
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()