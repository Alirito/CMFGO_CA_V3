import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from timm.data.random_erasing import RandomErasing 
import torch.distributed as dist
# from datasets.sampler import RandomIdentitySampler
from .sampler import RandomIdentitySampler
from .sampler_ddp import RandomIdentitySampler_DDP

# from sampler import RandomIdentitySampler
# from sampler_ddp import RandomIdentitySampler_DDP

# 从当前包导入基础图像数据集类
from .bases import ImageDataset
# from bases import ImageDataset

# 从当前包导入HOSS数据集类
# from .hoss import HOSS
from .cmfgo import CMFGO
# from cmfgo import CMFGO

import argparse


# 创建数据集名称到数据集类的映射字典
__factory = {
    # "HOSS": HOSS,
    "CMFGO": CMFGO
}

# 定义训练数据整理函数
# 训练时：不需要图像路径，只关心图像内容和标签
# 所有标签都需要张量化，因为要参与损失计算和梯度反向传播
def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    # 解压batch数据，获取图像、ID、摄像头ID、视角ID、图像路径（用_忽略）、图像尺寸
    imgs, pids, camids, viewids, _, img_size = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    img_size = torch.tensor(img_size, dtype=torch.float32)
    # 返回堆叠的图像张量和其他标签张量
    return torch.stack(imgs, dim=0), pids, camids, viewids, img_size


def train_pair_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    rgb_batch = [i[0] for i in batch]
    sar_batch = [i[1] for i in batch]
    batch = rgb_batch + sar_batch
    imgs, pids, camids, _, _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids


# 验证时：需要图像路径进行结果分析、可视化、错误分析等
# 验证时：PID不需要张量化，因为验证阶段主要是特征提取和相似度计算，不需要计算分类损失
def val_collate_fn(batch):
    # 解压batch数据，这次包含图像路径
    imgs, pids, camids, viewids, img_paths, img_size = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    img_size = torch.tensor(img_size, dtype=torch.float32)
    return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths, img_size


def make_dataloader(cfg):
    train_transforms = T.Compose(
        [
            # 创建训练数据变换流水线，首先调整图像大小，使用插值方法3（双三次插值）
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            
            # 随机水平翻转，概率从配置读取
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            
            # 图像填充
            T.Pad(cfg.INPUT.PADDING),
            
            # 随机裁剪到训练尺寸
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            
            # 将PIL图像转换为Tensor
            T.ToTensor(),
            
            # 图像归一化，使用配置的均值和标准差
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            
            # 随机擦除数据增强，像素模式，最大计数1，使用CPU设备
            RandomErasing(probability=cfg.INPUT.RE_PROB, mode="pixel", max_count=1, device="cpu"),
            # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ]
    )

    val_transforms = T.Compose(
        [
            # 创建验证变换，调整到测试尺寸
            T.Resize(cfg.INPUT.SIZE_TEST),
            
            # 将PIL图像转换为Tensor
            T.ToTensor(),
            
            # 图像归一化，使用配置的均值和标准差
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
        ]
    )

    # 从配置获取数据加载工作进程数
    num_workers = cfg.DATALOADER.NUM_WORKERS

    # 使用工厂模式创建数据集实例
    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)

    # 创建训练集，应用训练变换
    train_set = ImageDataset(dataset.train, train_transforms)
    # 创建另一个训练集(无数据增强的数据集)，但使用验证变换(用于特征提取) ———— 为什么使用不同变换？
    train_set_normal = ImageDataset(dataset.train, val_transforms)
    # 获取训练集中的人物ID数量（类别数）
    num_classes = dataset.num_train_pids
    # 获取训练集中的摄像头数量 
    cam_num = dataset.num_train_cams

    # 检查是否使用三元组采样器
    if "triplet" in cfg.DATALOADER.SAMPLER:
        # 检查是否使用分布式训练
        if cfg.MODEL.DIST_TRAIN:
            print("DIST_TRAIN START")
            # 计算每个进程的批次大小
            mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // dist.get_world_size()
            # 创建分布式随机身份采样器
            data_sampler = RandomIdentitySampler_DDP(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
            # 创建批次采样器
            batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
            # 创建分布式训练数据加载器
            train_loader = DataLoader(
                train_set,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=train_collate_fn,
                pin_memory=True,
            )
        else:
            # 创建非分布式训练数据加载器
            train_loader = DataLoader(
                train_set,
                batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers,
                collate_fn=train_collate_fn,
            )
    # 检查是否使用softmax采样器
    elif cfg.DATALOADER.SAMPLER == "softmax":
        print("using softmax sampler")
        # 创建简单shuffle的数据加载器
        train_loader = DataLoader(
            train_set, 
            batch_size=cfg.SOLVER.IMS_PER_BATCH, 
            shuffle=True, 
            num_workers=num_workers, 
            collate_fn=train_collate_fn
        )
    else:
        # 处理不支持的采样器类型
        print("unsupported sampler! expected softmax or triplet but got {}".format(cfg.SAMPLER))

    # 创建验证集，合并查询集和画廊集  (画廊集？)
    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)

    # 创建验证数据加载器
    val_loader = DataLoader(
        val_set, 
        batch_size=cfg.TEST.IMS_PER_BATCH, 
        shuffle=False, 
        num_workers=num_workers, 
        collate_fn=val_collate_fn
    )
    
    # 创建用于特征提取的训练数据加载器（使用验证变换）
    train_loader_normal = DataLoader(
        train_set_normal, 
        # 使用测试批次大小
        batch_size=cfg.TEST.IMS_PER_BATCH, 
        # 不打乱顺序
        shuffle=False, 
        num_workers=num_workers, 
        # 使用验证整理函数，包含更多元数据（如图像路径）
        collate_fn=val_collate_fn
    )
    # 检查批次大小是否为偶数
    if cfg.SOLVER.IMS_PER_BATCH % 2 != 0:
        raise ValueError("cfg.SOLVER.IMS_PER_BATCH should be even number")
    return train_loader, train_loader_normal, val_loader, len(dataset.query), num_classes, cam_num


def make_dataloader_pair(cfg):
    train_transforms = T.Compose(
        [
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mode="pixel", max_count=1, device="cpu"),
            # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ]
    )

    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)

    train_set_pair = ImageDataset(dataset.train_pair, train_transforms, pair=True)
    num_classes = dataset.num_train_pair_pids
    cam_num = dataset.num_train_pair_cams

    if cfg.SOLVER.IMS_PER_BATCH % 2 != 0:
        raise ValueError("cfg.SOLVER.IMS_PER_BATCH should be even number")
    train_loader_pair = DataLoader(
        train_set_pair, 
        batch_size=int(cfg.SOLVER.IMS_PER_BATCH / 2), 
        shuffle=True, 
        num_workers=num_workers, 
        collate_fn=train_pair_collate_fn
    )
    return train_loader_pair, num_classes, cam_num


if __name__ == "__main__":
    from defaults import _C as cfg
    parser = argparse.ArgumentParser(description="TransOSS Training")
    parser.add_argument("--config_file", default="CMFGO/configs/cmfgo_transoss.yml", help="path to config file", type=str)
    # parser.add_argument("--config_file", default="../configs/pretrain_transoss.yml", help="path to config file", type=str)
    args = parser.parse_args()
    
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    # cfg.merge_from_list(args.opts)
    cfg.freeze()
    
    train_transforms = T.Compose(
        [
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mode="pixel", max_count=1, device="cpu"),
            # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ]
    )
    
    # print("=== 执行过程分析 ===")

    # # 1. 查看配置
    # print(f"要使用的数据集: {cfg.DATASETS.NAMES}")
    # print(f"数据集根目录: {cfg.DATASETS.ROOT_DIR}")
    
    # # 2. 工厂字典查找
    # dataset_class = __factory[cfg.DATASETS.NAMES]
    # print(f"找到的数据集类: {dataset_class}")
    
    # # 3. 实例化数据集
    # dataset = dataset_class(root=cfg.DATASETS.ROOT_DIR)
    # print("dataset", dataset)
    
    # # 4. 查看数据集内容
    # print(f"训练集样本数: {len(dataset.train)}")
    # print(f"训练集前2个样本: {dataset.train[:2]}")
    
    # # 5. 创建训练集
    # train_set_pair = ImageDataset(dataset.train_pair, train_transforms, pair=True)
    # # train_set = ImageDataset(dataset.train, train_transforms)
    # print(f"训练集包装完成，样本数: {len(train_set_pair)}")
    # print(f"训练集: {train_set_pair}")
    
    # # 测试获取一个样本
    # sample_img, sample_pid, sample_camid, _, _ = train_set_pair[0][0]
    # sample_img2, sample_pid2, sample_camid2, _, _ = train_set_pair[0][1]
    # print(f"第一1个样本: 图片={sample_img}, ID={sample_pid}, 摄像头={sample_camid}")
    # print(f"第一2个样本: 图片={sample_img2}, ID={sample_pid2}, 摄像头={sample_camid2}")
    
    print("=== 执行过程分析 ===")

    # 1. 查看配置
    print(f"要使用的数据集: {cfg.DATASETS.NAMES}")
    print(f"数据集根目录: {cfg.DATASETS.ROOT_DIR}")
    
    # 2. 工厂字典查找
    dataset_class = __factory[cfg.DATASETS.NAMES]
    print(f"找到的数据集类: {dataset_class}")
    
    # 3. 实例化数据集
    dataset = dataset_class(root=cfg.DATASETS.ROOT_DIR)
    print("dataset", dataset)
    
    # 4. 查看数据集内容
    print(f"查询集样本数: {len(dataset.query)}")
    print(f"查询集前2个样本: {dataset.query[:2]}")
    print(f"画廊集样本数: {len(dataset.gallery)}")
    print(f"画廊集前2个样本: {dataset.gallery[:2]}")
    
    # 5. 创建验证集
    val_set = ImageDataset(dataset.query + dataset.gallery, train_transforms)
    print(f"验证集包装完成，样本数: {len(val_set)}")
    print(f"验证集: {val_set.dataset}")
    
    # # 测试获取一个样本
    # sample_img, sample_pid, sample_camid, _, _ = train_set_pair[0][0]
    # sample_img2, sample_pid2, sample_camid2, _, _ = train_set_pair[0][1]
    # print(f"第一1个样本: 图片={sample_img}, ID={sample_pid}, 摄像头={sample_camid}")
    # print(f"第一2个样本: 图片={sample_img2}, ID={sample_pid2}, 摄像头={sample_camid2}")
    
    # num_workers = cfg.DATALOADER.NUM_WORKERS
    # val_loader = DataLoader(
    #     val_set, 
    #     batch_size=cfg.TEST.IMS_PER_BATCH, 
    #     shuffle=False, 
    #     num_workers=num_workers, 
    #     collate_fn=val_collate_fn
    # )
    # print("\n\n\n=== 读取一个验证批次样本 ===")
    # for n_iter, (img, vid, camid, camids, target_view, _, img_wh) in enumerate(val_loader):
    #     print("img:", img)
    #     print("vid:", vid)
    #     print("camid:", camid)
    #     print("camids:", camids)
    #     print("target_view:", target_view)
    #     print("img_wh:", img_wh)
    #     break
    