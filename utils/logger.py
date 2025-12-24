import logging
import os
import sys
import os.path as osp
def setup_logger(name, save_dir, if_train):
    # 根据给定的name获取或创建一个日志记录器实例
    logger = logging.getLogger(name)
    # 设置日志记录级别为DEBUG（最低级别，记录所有日志）
    logger.setLevel(logging.DEBUG)

    # 创建输出到标准输出的流处理器
    ch = logging.StreamHandler(stream=sys.stdout)
    # 设置处理器级别为DEBUG
    ch.setLevel(logging.DEBUG)
    # 创建日志格式器，定义日志输出格式
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    # 将格式器应用于控制台处理器
    ch.setFormatter(formatter)
    # 将处理器添加到日志记录器
    logger.addHandler(ch)

    # 检查是否提供了保存目录
    if save_dir:
        # 如果目录不存在则创建
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        # 根据if_train参数决定创建训练或测试日志文件，文件模式为'w'（写入模式，会覆盖原有文件）
        if if_train:
            fh = logging.FileHandler(os.path.join(save_dir, "train_log.txt"), mode='w')
        else:
            fh = logging.FileHandler(os.path.join(save_dir, "test_log.txt"), mode='w')
        # 设置文件处理器级别和格式器
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        # 将文件处理器添加到日志记录器
        logger.addHandler(fh)

    return logger