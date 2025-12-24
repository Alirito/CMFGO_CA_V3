from PIL import Image
from torch.utils.data import Dataset
import os.path as osp
import numpy as np
import cv2


# 定义读取图像的函数 —— 持续尝试读取图像直到成功
def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    # 初始化标志变量，表示是否成功读取图像
    got_img = False
    # 检查图像文件是否存在
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    # 循环直到成功读取图像
    while not got_img:
        try:
            # 使用PIL打开图像文件
            img = Image.open(img_path)
            # 设置成功标志为True
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        """
        获取数据集的基本统计信息
        data: 包含图像信息的列表，每个元素格式为 (图像路径, 人物ID, 摄像头ID, 轨迹ID)
        """
        pids, cams, tracks = [], [], []
        for _, pid, camid, trackid in data:
            pids += [pid]
            cams += [camid]
            tracks += [trackid]
        # 去重
        pids = set(pids)
        cams = set(cams)
        tracks = set(tracks)
        # 计算数量
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        num_views = len(tracks)
        return num_pids, num_imgs, num_cams, num_views

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        """
            打印数据集的详细统计信息
            train: 训练集数据
            query: 查询集数据  
            gallery: 图库集数据
        """
        if train is not None:
            num_train_pids, num_train_imgs, num_train_cams, num_train_views = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams, num_train_views = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_train_views = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        if train is not None:
            print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")


# 定义SAR图像转换函数
def sar32bit2RGB(img):
    # 将PIL图像转换为float32类型的NumPy数组
    nimg = np.array(img, dtype=np.float32)
    # 将图像数据归一化到0-255范围
    nimg = nimg / nimg.max() * 255
    # 将数据类型转换为8位无符号整数
    nimg_8 = nimg.astype(np.uint8)
    # 使用OpenCV将灰度图转换为RGB三通道图像
    # print("nimg_8 shape: ", nimg_8.shape)
    # print("nimg_8 shape[2]: ", nimg_8.shape[2])
    
    if(len(nimg_8.shape) !=2):
        pil_img = Image.fromarray(nimg_8)
        return pil_img
    
    cv_img = cv2.cvtColor(nimg_8, cv2.COLOR_GRAY2RGB)
    # 将NumPy数组转换回PIL图像格式
    pil_img = Image.fromarray(cv_img)
    return pil_img


# 定义ImageDataset类，继承自PyTorch的Dataset类
class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None, pair=False):
        # 保存数据集引用
        self.dataset = dataset
        # 保存图像变换函数
        self.transform = transform
        # 保存pair标志，表示是否处理图像对
        self.pair = pair

    # 定义返回数据集长度的方法
    def __len__(self):
        return len(self.dataset)
    
    # 定义获取单个图像的方法
    def get_image(self, img_path):
        # 检查是否为SAR图像（文件扩展名为SAR.tif）
        if img_path.endswith("SAR.tif"):
            # 读取SAR图像
            # print("Reading SAR image: ", img_path)
            img = read_image(img_path)
            # 将32位SAR图像转换为RGB格式
            img = sar32bit2RGB(img)
            # 获取图像尺寸（宽度, 高度）
            img_size = img.size
        # 如果不是SAR图像（光学图像）
        else:
            # 读取图像并转换为RGB模式
            img = read_image(img_path).convert("RGB")
            # 获取图像尺寸
            img_size = img.size
            # 将图像尺寸缩放为原来的0.75倍（这里可能用于尺寸标准化）
            img_size = [img_size[0] * 0.75, img_size[1] * 0.75]
        # 对图像尺寸进行复杂的归一化计算，生成3个特征值
        # 需要修改！！！！！
        # 这些魔法数字（93, 427, 0.434, 0.461, 0.031）很可能是从训练数据统计得到的：
        img_size = (
            (img_size[0] / 93 - 0.434) / 0.031, 
            (img_size[1] / 427 - 0.461) / 0.031, 
            img_size[1] / img_size[0])
        # 检查是否定义了图像变换
        if self.transform is not None:
            # 应用图像变换（如缩放、裁剪、归一化等）
            img = self.transform(img)
        # 返回处理后的图像和尺寸特征
        return img, img_size

    # 定义获取数据样本的方法
    def __getitem__(self, index):
        # 检查是否处理图像对
        if self.pair:
            # 初始化图像列表
            imgs = []
            # 遍历当前索引对应的图像对列表
            for img in self.dataset[index]:
                # 解包图像路径、人物ID、摄像头ID
                img_path, pid, camid = img
                # 调用get_image方法获取图像和尺寸特征
                im, img_size = self.get_image(img_path)
                # 将图像数据添加到列表，包含图像、PID、摄像头ID、文件名、尺寸特征
                imgs.append((im, pid, camid, img_path.split("/")[-1], img_size))
            # 返回图像对列表
            return imgs
        # 如果不是图像对模式
        else:
            # 解包单个图像的数据
            img_path, pid, camid, trackid = self.dataset[index]
            img = self.dataset[index]
            # 调用get_image方法获取图像和尺寸特征
            img, img_size = self.get_image(img_path)
            # 返回单个图像的数据
            return img, pid, camid, trackid, img_path.split("/")[-1], img_size