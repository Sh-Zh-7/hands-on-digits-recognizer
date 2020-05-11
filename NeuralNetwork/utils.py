import os
import json
import shutil
import logging
import torch

class Params:
    def __init__(self, json_path):
        with open(json_path, "r") as f:
            self.__dict__.update(json.load(f))

    # We all need json path while loading and saving parameters
    def save(self, json_path):
        with open(json_path, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    def load(self, json_path):
        with open(json_path, "r") as f:
            self.__dict__.update(json.load(f))

    @property
    def dict(self):
        return self.__dict__

class RunningAverage:
    def __init__(self):
        self.total = 0
        self.count = 0

    def update(self, val):
        self.total += val
        self.count += 1

    def __call__(self):
        return self.total / self.count

def SetLogger(log_path):
    """ Decide which directory to log """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.handlers:
        # Add file handler
        file_handler = logging.FileHandler(os.path.join(log_path, "train.log"))
        file_handler.setFormatter(logging.Formatter("%(asctime)s:%(levelname)s: %(message)s"))
        logger.addHandler(file_handler)
        # Add stream handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)

def SaveDictToJson(d: dict, path):
    # 这里是专门用来保存metrics的
    # d中的元素转为python中的float然后进行dump
    new_dict = {
        k: float(v) for k, v in d.items()
    }
    with open(path, "w") as file:
        json.dump(new_dict, file, indent=4)

def SaveCheckPoint(state, checkpoint, is_best):
    # 判断checkpoint是否存在(注意checkpoint是一个目录)
    # 这里checkpoint得是绝对路径了（不然你还想在里面把绝对路径补全吗）
    # 每一个epoch保存一个last.pth.tar和best.pth.tar。（当然最后也只有这一个）
    if not os.path.exists(checkpoint):
        print("Checkpoint doesn't exist!Make directory named {}".format(checkpoint))
        os.makedirs(checkpoint)
    else:
        print("Checkpoint already exist!")

    file_path = os.path.join(checkpoint, "last.pth.tar")
    # 注意这里保存的是一个字典，包括epoch, model_state_dict, optimizer_state_dict
    torch.save(state, file_path)
    if is_best:
        shutil.copyfile(file_path, os.path.join(checkpoint, "best.pth.tar"))

def LoadCheckPoint(checkpoint, model, optimizer=None):
    if not os.path.exists(checkpoint):
        # 区别print warning和raise异常的区别
        # 前者还可以继续执行，后者是直接中断了
        raise("File doesn't exist!{}".format(checkpoint))

    # 你要知道python参数传进来的都是引用
    # 所以这里相当于是对原来的对象进行直接的修改
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint["state_dict"])

    if optimizer:
        optimizer.load_state_dict(checkpoint["optim_dict"])

    return checkpoint
