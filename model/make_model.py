import torch
import torch.nn as nn
from .backbones.vit_transoss import vit_base_patch16_224_TransOSS


# 对不同层类型（Linear、Conv、BatchNorm）分别初始化权重
# 参数m是神经网络模块
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


# 分类器的特殊初始化，使用较小的标准差
def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class build_transformer(nn.Module):
    # 继承nn.Module
    def __init__(self, num_classes, camera_num, cfg, factory, logit_scale_init_value=2.6592):
        # 接收类别数、摄像头数、配置对象、模型工厂等参数
        super(build_transformer, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE             # 从配置中获取最后层的步长
        model_path = cfg.MODEL.PRETRAIN_PATH            # 从配置中获取预训练模型路径
        model_name = cfg.MODEL.NAME                     # 从配置中获取模型名称
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE     # 从配置中获取预训练选择
        self.cos_layer = cfg.MODEL.COS_LAYER            # 设置是否使用cosine层
        self.neck = cfg.MODEL.NECK                      # 设置neck结构类型
        self.neck_feat = cfg.TEST.NECK_FEAT             # 设置测试时使用的neck特征类型
        self.in_planes = 768                            # 设置输入平面数为768（ViT-base的标准维度）
        self.model_type = cfg.MODEL.TRANSFORMER_TYPE # 设置模型类型
        self.camera_num = camera_num
        
        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))
            
        # 检查是否使用特定的ViT变体
        if cfg.MODEL.TRANSFORMER_TYPE == 'vit_base_patch16_224_TransOSS':
            self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size = cfg.INPUT.SIZE_TRAIN, 
                                                            stride_size = cfg.MODEL.STRIDE_SIZE, 
                                                            camera = camera_num,
                                                            drop_path_rate = cfg.MODEL.DROP_PATH,
                                                            drop_rate = cfg.MODEL.DROP_OUT,
                                                            attn_drop_rate = cfg.MODEL.ATT_DROP_RATE)
        else:
            # 如果模型类型不支持，抛出错误
            raise ValueError('Unsupported model type: {}'.format(cfg.MODEL.TRANSFORMER_TYPE))

        # 保存类别数
        self.num_classes = num_classes
        # 保存ID损失类型
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        
        # 分类器层和特征瓶颈层
        # 创建分类器线性层，输入768维，输出类别数个维度，无偏置
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        # 对分类器应用分类器特定的权重初始化
        self.classifier.apply(weights_init_classifier)

        # 瓶颈层使用BatchNorm并固定bias
        # 创建1D批量归一化层作为瓶颈层
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        # 设置瓶颈层的偏置项不需要梯度更新
        self.bottleneck.bias.requires_grad_(False)
        # 对瓶颈层应用Kaiming权重初始化
        self.bottleneck.apply(weights_init_kaiming)

        # train_pair控制是否进行跨模态对训练
        # 初始化train_pair标志为False
        self.train_pair = False
        # 创建可学习的logit缩放参数
        self.logit_scale = nn.Parameter(torch.tensor(logit_scale_init_value))

        # 预训练权重加载 —— 支持多种预训练方式
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained model from {}'.format(model_path))
        elif pretrain_choice == 'clip':
            self.load_param(model_path)
            print('Loading pretrained model from {}'.format(model_path))
        elif pretrain_choice == False:
            print('Training transformer from scratch.')
        else:
            raise ValueError('Unsupported pretrain_choice: {}'.format(pretrain_choice))

    def train_with_pair(self,):
        self.train_pair = True

    def train_with_single(self,):
        self.train_pair = False

# 定义前向传播方法，参数包括输入x、标签、摄像头标签、图像宽高
    def forward(self, x, label=None, cam_label=None):
        # 通过基础Transformer获取全局特征和预测掩码 (OSSDet 改进：接收掩码)
        global_feat, pred_masks = self.base(x, cam_label=cam_label)
        if self.training:
            if self.train_pair:  # 跨模态对训练
                # 获取batch size
                b_s = global_feat.size(0)
                # normalized features
                # 分割光学和SAR图像特征
                opt_embeds = global_feat[0:b_s // 2]    # 取前一半batch作为光学图像嵌入
                sar_embeds = global_feat[b_s // 2:]     # 取后一半batch作为SAR图像嵌入
                
                # 特征归一化
                # 对光学嵌入进行L2归一化
                opt_embeds = opt_embeds / opt_embeds.norm(p=2, dim=-1, keepdim=True)
                # 对SAR嵌入进行L2归一化
                sar_embeds = sar_embeds / sar_embeds.norm(p=2, dim=-1, keepdim=True)

                # 计算跨模态相似度
                # 计算可学习的logit缩放值
                logit_scale = self.logit_scale.exp()
                # 计算SAR到光学的相似度矩阵并缩放
                logits_per_sar = torch.matmul(sar_embeds, opt_embeds.t()) * logit_scale
                # 返回相似度矩阵
                return logits_per_sar

            else:   # 单模态训练
                # 通过瓶颈层处理特征
                feat = self.bottleneck(global_feat)
                # 检查是否使用特定的度量学习损失
                if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                    # 使用带标签的分类器
                    cls_score = self.classifier(feat, label)    # 使用度量学习损失
                else:
                    # 使用普通分类器
                    cls_score = self.classifier(feat)   # 普通分类

                # 返回分类分数、全局特征以及预测掩码 (用于 Act Loss)
                return cls_score, global_feat, pred_masks  # global feature for triplet loss
        
        else:   # 测试模式
            # 检查是否使用瓶颈层后的特征
            if self.neck_feat == 'after':
                # 通过瓶颈层处理特征
                feat = self.bottleneck(global_feat)
                # print("Test with feature after BN")
                return feat  # 使用 BN 后的特征
            else:
                # print("Test with feature before BN")
                # 返回原始全局特征
                return global_feat  # 使用原始特征

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if "state_dict" in param_dict:
            param_dict = param_dict["state_dict"]
        # print(self.state_dict().keys())
        for i in param_dict:
            key = i.replace('module.', '')
            # 跳过分类器参数
            if key.startswith('classifier'):
                continue
            self.state_dict()[key].copy_(param_dict[i])

    # 定义微调时加载参数的方法
    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

# 定义模型类型工厂字典
__factory_T_type = {
    'vit_base_patch16_224_TransOSS': vit_base_patch16_224_TransOSS,
}

def make_model(cfg, num_class, camera_num):
    model = build_transformer(num_class, camera_num, cfg, __factory_T_type)
    print('===========building transformer===========')
    return model