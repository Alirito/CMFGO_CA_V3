from torch.utils.data.sampler import Sampler
from collections import defaultdict
import copy
import random
import numpy as np


# 定义随机身份采样器类，继承自Sampler基类
class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    # 定义初始化方法，接收数据源、批次大小、每个身份的实例数
    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        # 保存每个身份的实例数
        self.num_instances = num_instances
        # 计算每个批次中包含的身份数 N = batch_size / num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        # 创建默认字典，用于按身份ID分组存储数据索引   ？为什么按照身份ID分组？身份ID有什么用？
        self.index_dic = defaultdict(list)
        # 遍历数据源，index是数据索引，pid是身份ID
        # 举个例子说明一下这段代码的含义？
        for index, (_, pid, _, _) in enumerate(self.data_source):
            # 将每个数据的索引添加到对应身份ID的列表中
            self.index_dic[pid].append(index)
        # 获取所有不重复的身份ID列表
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        # 估计一个epoch中的样本数量
        self.length = 0
        for pid in self.pids:
            # 获取该身份对应的所有数据索引
            idxs = self.index_dic[pid]
            # 计算该身份拥有的样本数量
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    # 迭代器方法 __iter__
    def __iter__(self):
        # 创建批次索引字典，按身份分组存储批次
        batch_idxs_dict = defaultdict(list)

        # 遍历每个身份ID
        for pid in self.pids:
            # 深拷贝该身份的所有数据索引（避免修改原始数据）
            idxs = copy.deepcopy(self.index_dic[pid])
            # 如果该身份的样本数少于需要的实例数
            if len(idxs) < self.num_instances:
                # 使用有放回抽样补足到 num_instances 个样本，重复放回
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            # 打乱该身份的样本顺序
            random.shuffle(idxs)
            # 创建临时列表存储当前批次的索引
            batch_idxs = []
            # 遍历打乱后的索引
            for idx in idxs:
                # 将索引添加到当前批次
                batch_idxs.append(idx)
                # 如果当前批次已满（达到num_instances个）
                
                # 当batch_idxs长度达到2时：
                # 第一次: batch_idxs = [1,2] → 添加到批次字典
                # 第二次: batch_idxs = [0] → 长度不足2，继续
                if len(batch_idxs) == self.num_instances:
                    # 将这个完整批次添加到该身份的批次列表中
                    batch_idxs_dict[pid].append(batch_idxs)
                    # 重置当前批次列表
                    batch_idxs = []

        # 深拷贝可用的身份ID列表
        avai_pids = copy.deepcopy(self.pids)
        # 创建最终索引列表
        final_idxs = []

        # 当还有足够身份可以组成一个完整批次时循环
        while len(avai_pids) >= self.num_pids_per_batch:
            # 随机选择num_pids_per_batch个身份
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            # 遍历每个选中的身份
            for pid in selected_pids:
                # 从该身份的批次列表中取出第一个批次
                batch_idxs = batch_idxs_dict[pid].pop(0)
                # 将这个批次的索引扩展到最终列表中
                final_idxs.extend(batch_idxs)
                # 如果该身份没有剩余批次了
                if len(batch_idxs_dict[pid]) == 0:
                    # 从可用身份列表中移除该身份
                    avai_pids.remove(pid)

        # 返回最终索引列表的迭代器
        return iter(final_idxs)

    def __len__(self):
        return self.length


class BalancedClassSampler(Sampler):
    """平衡类别采样器，确保每个batch中各类别样本数量平衡"""
    def __init__(self, data_source, batch_size, samples_per_class):
        """
        Args:
            data_source: 数据集，每个样本是(img_path, class_id, ...)
            batch_size: 批次大小
            samples_per_class: 每个类别的样本数
        """
        self.data_source = data_source
        self.batch_size = batch_size
        self.samples_per_class = samples_per_class
        
        # 按类别分组
        self.class_indices = {}
        for idx, (_, class_id, *_) in enumerate(data_source):
            if class_id not in self.class_indices:
                self.class_indices[class_id] = []
            self.class_indices[class_id].append(idx)
            
        self.classes = list(self.class_indices.keys())
        self.num_classes = len(self.classes)
        
    def __iter__(self):
        """生成索引迭代器"""
        indices = []
        
        # 计算需要多少个类别
        num_classes_in_batch = self.batch_size // self.samples_per_class
        
        # 确保有足够类别
        if num_classes_in_batch > self.num_classes:
            num_classes_in_batch = self.num_classes
            
        # 生成多个批次
        for _ in range(len(self.data_source) // self.batch_size + 1):
            # 随机选择类别
            selected_classes = random.sample(self.classes, min(num_classes_in_batch, self.num_classes))
            
            for class_id in selected_classes:
                class_indices = self.class_indices[class_id]
                
                # 从该类别中采样
                if len(class_indices) >= self.samples_per_class:
                    sampled = random.sample(class_indices, self.samples_per_class)
                else:
                    # 如果样本不够，允许重复采样
                    sampled = random.choices(class_indices, k=self.samples_per_class)
                    
                indices.extend(sampled)
                
        return iter(indices[:len(self.data_source)])
    
    def __len__(self):
        return len(self.data_source)