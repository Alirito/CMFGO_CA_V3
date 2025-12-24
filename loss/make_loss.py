# encoding: utf-8
"""
@author:  Xiuwei Wang
@contact: w694309396@163.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss


class ObjectActivationLoss(nn.Module):
    """ 
    借鉴 OSSDet 的对象激活损失 (Act loss) [cite: 774, 884]
    用于显式强制目标特定的激活，抑制背景噪声 [cite: 774, 885]。
    """
    def __init__(self, gamma=0.1, sigma=0.5):
        super(ObjectActivationLoss, self).__init__()
        # 修复点：确保 gamma 在初始化中定义 
        self.gamma = gamma # 控制前后景平衡的系数 
        self.sigma = sigma # 用于生成伪标签的高斯标准差

    def forward(self, pred_masks, device):
        total_act_loss = 0.0
        count = 0
        
        # pred_masks 是 backbone 返回的掩码元组 (mask_rgb, mask_sar)
        for mask in pred_masks:
            if mask is not None:
                B, _, H, W = mask.shape
                # 生成中心高斯伪标签 Mg [cite: 776]
                grid_y, grid_x = torch.meshgrid(
                    torch.linspace(-1, 1, H, device=device), 
                    torch.linspace(-1, 1, W, device=device)
                )
                dist = torch.exp(-(grid_x**2 + grid_y**2) / (2 * self.sigma**2))
                gt_mask = dist.expand(B, 1, H, W)
                
                # 计算 L_I (促进目标区域激活) [cite: 777, 778]
                loss_i = 1 - (mask * gt_mask).sum() / (gt_mask.sum() + 1e-6)
                # 计算 L_D (惩罚背景区域激活) [cite: 777, 778]
                bg_mask = 1 - (gt_mask > 0).float()
                loss_d = (mask * bg_mask).sum() / (mask.sum() + 1e-6)
                
                # 核心公式: L_act = L_I + gamma * L_D [cite: 779]
                total_act_loss += (loss_i + self.gamma * loss_d)
                count += 1
                
        return total_act_loss / count if count > 0 else torch.tensor(0.0, device=device)


def make_loss(cfg, num_classes):  # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    # 设置特征维度为2048
    feat_dim = 2048
    # 创建中心损失实例，用于拉近同类样本的特征
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    
    # OSSDet 改进：创建对象激活损失实例，gamma 推荐值为 0.1 
    act_criterion = ObjectActivationLoss(gamma=0.1)
    
    # 根据配置创建三元组损失，支持有无margin的版本
    if "triplet" in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    # 如果度量损失类型不是triplet，打印警告信息
    else:
        print("expected METRIC_LOSS_TYPE should be triplet" "but got {}".format(cfg.MODEL.METRIC_LOSS_TYPE))

    # 如果开启标签平滑，创建标签平滑的交叉熵损失
    if cfg.MODEL.IF_LABELSMOOTH == "on":
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, num_classes: ", num_classes)

    # 如果是softmax采样器，定义只使用交叉熵损失的函数
    if sampler == "softmax":

        # 定义损失函数（接收分数、特征、目标）
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)

    # 如果是softmax+triplet采样器，进入复合损失分支
    elif sampler == "softmax_triplet":

        # 定义损失函数（修复点：参数列表增加 pred_masks=None）
        def loss_func(score, feat, target, target_cam, pred_masks=None):
            # 检查度量损失类型是否为triplet
            if cfg.MODEL.METRIC_LOSS_TYPE == "triplet":
                # 检查是否开启标签平滑
                if cfg.MODEL.IF_LABELSMOOTH == "on":
                    # 检查score是否为列表类型
                    if isinstance(score, list):
                        # 对score列表中除第一个外的所有元素计算标签平滑损失
                        ID_LOSS = [xent(scor, target) for scor in score[1:]]
                        # 计算平均ID损失
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        # 加权组合ID损失
                        ID_LOSS = 0.5 * ID_LOSS + 0.5 * xent(score[0], target)
                    
                    # else分支开始（score不是列表）
                    else:
                        # 直接计算ID损失   ？？怎么计算
                        ID_LOSS = xent(score, target)

                    # 检查feat是否为列表类型
                    if isinstance(feat, list):
                        # 对feat列表中除第一个外的所有元素计算三元组损失
                        TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                        # 计算平均三元组损失
                        TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                        # 加权组合三元组损失
                        TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                    else:
                        # 直接计算三元组损失
                        TRI_LOSS = triplet(feat, target)[0]

                    # 修复点：显式定义 base_loss，再进行 OSSDet 权重的加法 
                    base_loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                    
                    # 检查是否存在掩码，若存在则加入对象激活损失，alpha 推荐值为 0.6 
                    if pred_masks is not None:
                        act_loss = act_criterion(pred_masks, target.device)
                        return base_loss + 0.6 * act_loss # 0.6 为论文推荐的 alpha 权重 [cite: 881]
                        
                    return base_loss
                    
                # else分支开始（标签平滑关闭）
                else:
                    # 检查score是否为列表类型
                    if isinstance(score, list):
                        # 对score列表中除第一个外的所有元素计算标准交叉熵损失
                        ID_LOSS = [F.cross_entropy(scor, target) for scor in score[1:]]
                        # 计算平均ID损失
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        # 加权组合ID损失
                        ID_LOSS = 0.5 * ID_LOSS + 0.5 * F.cross_entropy(score[0], target)
                    # else分支开始（score不是列表）
                    else:
                        # 直接计算ID损失
                        ID_LOSS = F.cross_entropy(score, target)

                    # 检查feat是否为列表类型
                    if isinstance(feat, list):
                        # 对feat列表中除第一个外的所有元素计算三元组损失
                        TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                        # 计算平均三元组损失
                        TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                        # 加权组合三元组损失
                        TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                    else:
                        # 直接计算三元组损失
                        TRI_LOSS = triplet(feat, target)[0]

                    # 修复点：同上 
                    base_loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                    
                    if pred_masks is not None:
                        act_loss = act_criterion(pred_masks, target.device)
                        return base_loss + 0.6 * act_loss 
                        
                    return base_loss
            else:
                print("expected METRIC_LOSS_TYPE should be triplet" "but got {}".format(cfg.MODEL.METRIC_LOSS_TYPE))

    else:
        print("expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center" "but got {}".format(cfg.DATALOADER.SAMPLER))
    return loss_func, center_criterion