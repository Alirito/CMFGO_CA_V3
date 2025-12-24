import torch
from torch.utils.data.sampler import Sampler
import torch.distributed as dist
from collections import defaultdict
import math
import copy
import numpy as np

import pickle


def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    """
    if dist.get_backend() == "nccl":
        return dist.new_group(backend="gloo")
    else:
        return dist.group.WORLD


def _pad_to_largest_tensor(tensor, group):
    """
    将所有rank的tensor填充到最大尺寸以便all_gather操作
    Returns:
        list[int]: size of the tensor, on each rank
        Tensor: padded tensor that has the max size
    """
    # 获取组内进程数
    world_size = dist.get_world_size(group=group)
    assert world_size >= 1, "comm.gather/all_gather must be called from ranks within the given group!"
    # 获取本地tensor的大小
    local_size = torch.tensor([tensor.numel()], dtype=torch.int64, device=tensor.device)
    # 准备接收所有rank的大小
    size_list = [torch.zeros([1], dtype=torch.int64, device=tensor.device) for _ in range(world_size)]
    # 收集所有rank的tensor大小
    dist.all_gather(size_list, local_size, group=group)
    # 转换为python int列表
    size_list = [int(size.item()) for size in size_list]
    # 找到最大尺寸
    max_size = max(size_list)

    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    # 如果本地tensor小于最大尺寸，进行填充
    if local_size != max_size:
        padding = torch.zeros((max_size - local_size,), dtype=torch.uint8, device=tensor.device)
        tensor = torch.cat((tensor, padding), dim=0)
    return size_list, tensor


def _serialize_to_tensor(data, group):
    """
        将任意picklable数据序列化为tensor
    """
    backend = dist.get_backend(group)
    assert backend in ["gloo", "nccl"]
    device = torch.device("cpu" if backend == "gloo" else "cuda")

    buffer = pickle.dumps(data)
    if len(buffer) > 1024**3:
        print(
            "Rank {} trying to all-gather {:.2f} GB of data on device {}".format(
                dist.get_rank(), len(buffer) / (1024**3), device
            )
        )
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device=device)
    return tensor


def all_gather(data, group=None):
    """
    在任意picklable数据上执行all_gather操作
    Run all_gather on arbitrary picklable data (not necessarily tensors).
    Args:
        data: any picklable object
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.
    Returns:
        list[data]: list of data gathered from each rank
    """
    # 单进程情况
    if dist.get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    if dist.get_world_size(group) == 1:
        return [data]
    # 序列化数据
    tensor = _serialize_to_tensor(data, group)
    
    # 填充到最大尺寸
    size_list, tensor = _pad_to_largest_tensor(tensor, group)
    max_size = max(size_list)

    # receiving Tensor from all ranks
    tensor_list = [torch.empty((max_size,), dtype=torch.uint8, device=tensor.device) for _ in size_list]
    dist.all_gather(tensor_list, tensor, group=group)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def shared_random_seed():
    """
    Returns:
        int: a random number that is the same across all workers.
            If workers need a shared RNG, they can use this shared seed to
            create one.
    All workers must call this function, otherwise it will deadlock.
    """
    ints = np.random.randint(2**31)
    all_ints = all_gather(ints)
    return all_ints[0]


class RandomIdentitySampler_DDP(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        # 进程总数
        self.world_size = dist.get_world_size() 
        self.num_instances = num_instances
        # 每个进程的batch大小
        self.mini_batch_size = self.batch_size // self.world_size   
        # 每个进程每个batch的身份数
        self.num_pids_per_batch = self.mini_batch_size // self.num_instances    
        # 按身份ID组织索引
        self.index_dic = defaultdict(list)

        # 构建索引字典：{pid: [index1, index2, ...]}
        for index, (_, pid, _, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)

        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        # 估算epoch中的样本数
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            # 确保每个身份的数量是num_instances的倍数
            self.length += num - num % self.num_instances

        # 当前进程的rank
        self.rank = dist.get_rank()
        # 每个进程的长度
        self.length //= self.world_size

    def __iter__(self):
        # 获取共享随机种子
        seed = shared_random_seed()
        # 设置种子确保所有进程采样顺序一致
        np.random.seed(seed)
        self._seed = int(seed)
        # 生成整个数据集的采样序列
        final_idxs = self.sample_list()

        # 计算每个进程应得的样本数
        length = int(math.ceil(len(final_idxs) * 1.0 / self.world_size))
        # final_idxs = final_idxs[self.rank * length:(self.rank + 1) * length]
        # 获取当前进程的索引
        final_idxs = self.__fetch_current_node_idxs(final_idxs, length)
        # 更新实际长度
        self.length = len(final_idxs)   
        return iter(final_idxs)

    def __fetch_current_node_idxs(self, final_idxs, length):
        # 总样本数
        total_num = len(final_idxs)
        # 每个进程能分到的小批量数
        block_num = length // self.mini_batch_size
        # 临时存储目标索引的列表
        index_target = []
        
        for i in range(0, block_num * self.world_size, self.world_size):
            # 为当前进程选择索引
            index = range(
                self.mini_batch_size * self.rank + self.mini_batch_size * i,
                min(self.mini_batch_size * self.rank + self.mini_batch_size * (i + 1), total_num),
            )
            index_target.extend(index)
        
        # 转换为numpy数组以便索引操作
        index_target_npy = np.array(index_target)
        
        # 从整个序列中选择当前进程的数据
        final_idxs = list(np.array(final_idxs)[index_target_npy])
        
        return final_idxs

    def sample_list(self):
        # np.random.seed(self._seed)
        # 可用身份ID列表
        avai_pids = copy.deepcopy(self.pids)
        # 存储每个身份的索引列表
        batch_idxs_dict = {}

        # 最终采样序列
        batch_indices = []
        # 当还有足够的身份时继续采样
        
        while len(avai_pids) >= self.num_pids_per_batch:
            # 随机选择num_pids_per_batch个身份
            selected_pids = np.random.choice(avai_pids, self.num_pids_per_batch, replace=False).tolist()
            for pid in selected_pids:
                if pid not in batch_idxs_dict:
                    idxs = copy.deepcopy(self.index_dic[pid])
                    # 如果某个身份的样本数不足，进行重复采样
                    if len(idxs) < self.num_instances:
                        idxs = np.random.choice(idxs, size=self.num_instances, replace=True).tolist()
                    np.random.shuffle(idxs)
                    batch_idxs_dict[pid] = idxs

                avai_idxs = batch_idxs_dict[pid]
                # 为每个身份取num_instances个样本
                for _ in range(self.num_instances):
                    batch_indices.append(avai_idxs.pop(0))

                # 如果某个身份的剩余样本不足，移除该身份
                if len(avai_idxs) < self.num_instances:
                    avai_pids.remove(pid)

        return batch_indices

    def __len__(self):
        return self.length