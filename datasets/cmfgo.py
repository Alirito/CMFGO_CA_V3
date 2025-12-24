# encoding: utf-8
import glob
import os.path as osp
from .bases import BaseImageDataset
# from bases import BaseImageDataset


# class HOSS(BaseImageDataset):
class CMFGO(BaseImageDataset):
    """
    CMFGO dataset
    """
    dataset_dir = 'CMFGO'

    def __init__(self, root='', verbose=True, pid_begin = 0, **kwargs):
    # def __init__(self, root='', verbose=True, pid_begin = "", **kwargs):
        """_summary_

        Args:
            root: 数据集根目录
            verbose: 是否打印详细信息
            pid_begin: 人物ID的起始编号(用于多个数据集合并时避免ID冲突)
        """
        super(CMFGO, self).__init__()
        # # 构建数据集各部分的完整路径
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        
        # training
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')
        
        # valation S2O
        # self.query_dir = osp.join(self.dataset_dir, 'subset/S2O/query')
        # self.gallery_dir = osp.join(self.dataset_dir, 'subset/S2O/bounding_box_test')
        
        # valation O2S
        # self.query_dir = osp.join(self.dataset_dir, 'subset/O2S/query')
        # self.gallery_dir = osp.join(self.dataset_dir, 'subset/O2S/bounding_box_test')

        self._check_before_run()
        self.pid_begin = pid_begin
        # 处理训练集（返回普通训练数据和配对数据）
        train, train_pair = self._process_dir_train(self.train_dir, relabel=True)
        # 处理查询集
        query = self._process_dir(self.query_dir, relabel=False)
        # 处理图库集
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> CMFGO ReID Dataset loaded")
            self.print_dataset_statistics(train, query, gallery)
            if train_pair is not None:
                print("Number of RGB-SAR pair: {}".format(len(train_pair)))
                print("  ----------------------------------------")

        self.train = train
        self.train_pair = train_pair
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)
        
        self.num_train_pair_pids, self.num_train_pair_imgs, self.num_train_pair_cams, self.num_train_pair_vids = self.get_imagedata_info_pair(self.train_pair)

    def get_imagedata_info_pair(self, data):
        """
            专门用于处理配对数据的统计信息方法
            data: 配对数据，每个元素是包含两个图像信息的列表
        """
        pids, cams, tracks = [], [], []

        for img in data:
            for _, pid, camid, trackid in img:
                pids += [pid]
                cams += [camid]
                tracks += [trackid]
        pids = set(pids)
        cams = set(cams)
        tracks = set(tracks)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        num_views = len(tracks)
        return num_pids, num_imgs, num_cams, num_views

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        """
            处理单个目录的数据(用于query和gallery)
            dir_path: 目录路径
            relabel: 是否重新标记人物ID(训练集需要，测试集不需要)
        """
        img_paths = glob.glob(osp.join(dir_path, '*.tif'))

        # 获取所有唯一的人物ID
        pid_container = set()
        for img_path in sorted(img_paths):
            # pid = int(img_path.split('/')[-1].split('_')[0])
            pid = img_path.split('/')[-1].split('_')[0]
            pid_container.add(pid)
        
        # 创建人物ID到标签的映射字典
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        # pid2label = {pid: pid for label, pid in enumerate(pid_container)}
        
        dataset = []
        for img_path in sorted(img_paths):
            # pid = int(img_path.split('/')[-1].split('_')[0])    # 提取人物ID
            pid = img_path.split('/')[-1].split('_')[0]     # 提取人物ID
            # camid 0 for RGB, 1 for SAR
            camid = 0 if img_path.split('/')[-1].split('_')[-1] == 'RGB.tif' else 1
            # if relabel: 
            #     pid = pid2label[pid]
            
            pid = pid2label[pid]
            
            # 添加到数据集：(图像路径, 人物ID, 摄像头ID, 轨迹ID)
            dataset.append((img_path, self.pid_begin + pid, camid, 1))
        return dataset

    def _process_dir_train(self, dir_path, relabel=False):
        """
            处理训练集目录, 除了普通处理外还生成RGB-SAR配对数据
        """
        img_paths = glob.glob(osp.join(dir_path, '*.tif'))

        # 分离出RGB图像路径
        RGB_paths = [i for i in img_paths if i.endswith('RGB.tif')]
        pid2sar = {} # 字典：人物ID -> 对应的SAR图像路径列表
        # pid2label = {pid: label for label, pid in enumerate(pid_container)}

        pid_container = set()
        for img_path in sorted(img_paths):
            # pid = int(img_path.split('/')[-1].split('_')[0])
            pid = img_path.split('/')[-1].split('_')[0]
            pid_container.add(pid)
            if img_path.endswith('SAR.tif'):
                # pid = pid2label[pid]
                if pid not in pid2sar:
                    pid2sar[pid] = [img_path]
                else:
                    pid2sar[pid].append(img_path)
        # 创建人物ID到标签的映射
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        # pid2label = {pid: pid for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in sorted(img_paths):
            # pid = int(img_path.split('/')[-1].split('_')[0])
            pid = img_path.split('/')[-1].split('_')[0]
            # camid 0 for RGB, 1 for SAR
            camid = 0 if img_path.split('/')[-1].split('_')[-1] == 'RGB.tif' else 1
            if relabel: 
                pid = pid2label[pid]
            dataset.append((img_path, self.pid_begin + pid, camid, 1))

        # 处理配对数据：每个RGB图像与对应的所有SAR图像配对
        dataset_pair = []
        for img_path in sorted(RGB_paths):
            # pid = int(img_path.split('/')[-1].split('_')[0])
            pid = img_path.split('/')[-1].split('_')[0]
            
            if pid not in pid2sar.keys():   # 如果没有对应的SAR图像，跳过
                continue
            for sar_path in pid2sar[pid]:   # 为每个RGB图像配对所有SAR图像
                temp_pid = pid2label[pid]
                dataset_pair.append([
                    (img_path, self.pid_begin + temp_pid, 0, 1), # RGB图像信息, 最后的1为轨迹ID，训练的时候用不到
                    (sar_path, self.pid_begin + temp_pid, 1, 1)  # SAR图像信息
                    ])

        return dataset, dataset_pair