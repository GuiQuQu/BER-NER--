import argparse
import logging

import torch
from pathlib import Path


def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train in the list:[cner]")
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="the input data dir,should include the training files")

    # parser.add_argument("--model_name_or_path", default=None, type=str, required=True)
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="the output dir where the model predictions and checkpoints will be written")

    parser.add_argument("--markup", default="bios", type=str, choices=["bios", "bio"])
    parser.add_argument("--train_max_seq_length", default=128, type=int)
    parser.add_argument("--eval_max_seq_length", default=512, type=int)


def init_logger(logger_name, logger_file="", logger_file_level=logging.NOTSET):
    if isinstance(logger_file, Path):
        logger_file = str(logger_file)
    log_format = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                                   datefmt="%m/%d/%Y %H:%M:%S %p")
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    if logger_file and logger_file != "":
        file_handler = logging.FileHandler(logger_file)
        file_handler.setFormatter(log_format)
        file_handler.setLevel(logger_file_level)
        logger.addHandler(file_handler)
    return logger


class CnerConfig(object):
    def __init__(self):
        # 基本设置
        self.model_type = "bert"
        self.task_name = "cner"
        self.data_dir = "dataset\\cner"
        self.output_dir = "outputs\\cner"
        self.markup = "bios"

        self.debug = True
        self.use_cuda = True
        self.device = "cuda:1" if torch.cuda.is_available() else "cpu"
        self.do_train = True
        self.do_eval = True
        self.do_predict = True
        self.overwrite_out_dir = True
        # 预处理设置
        self.overwrite_cache = False
        # 训练设置
        self.model_name_or_path = "bert-base-chinese"
        self.loss_type = "ce"
        self.batch_size = 32
        self.weight_decay = 0.01
        self.lr = 0.01
        self.adam_epsilon = 0.01
        self.warmup_step = 1000
        self.train_epochs = 3
        # bert max_seq_length
        self.train_max_seq_length = 128
        self.eval_max_seq_length = 512
        # 模型保存和评价
        self.save_step = -1
        self.dev_step = -1
