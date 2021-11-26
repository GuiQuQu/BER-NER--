import os

import random
import time
from torch.utils.tensorboard import SummaryWriter
import torch
from ner_metrics.ner_metrics import SeqEntityScore
import numpy as np
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import WEIGHTS_NAME, BertConfig, BertTokenizer
from model.bert_for_ner import BertSoftmaxForNer
from processors.ner_seq import collate_fn
from processors.ner_seq import processors, convert_examples_to_features
from tools.get_argparse import CnerConfig, init_logger
from torch.optim import AdamW

from torch.optim.lr_scheduler import LambdaLR

MODEL_CLASSES = {
    'bert': (BertConfig, BertSoftmaxForNer, BertTokenizer)
}

logger = init_logger("cner-logger", )


def seed_everything(seed=1029):
    """设置整个开发环境的seed,保证每一次运行随机数的随机结果都一样"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def train(config: CnerConfig, model, tokenizer):
    # 训练数据集和DataLoader
    processor = processors[config.task_name]()
    id2label = {i: label for i, label in enumerate(processor.get_labels())}
    # 训练数据
    train_dataset = load_and_cache_example(config, tokenizer, data_type="train")
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=config.batch_size, collate_fn=collate_fn)
    # 验证集数据
    eval_dataset = load_and_cache_example(config, tokenizer, data_type="dev")
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=config.batch_size,
                                 collate_fn=collate_fn)
    # 参数优化器(获取需要使用梯度下降计算的参数，学习率等超参数)
    no_decay = ["bias", "LayerNorm.weight"]
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': config.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(params=optimizer_grouped_parameters, lr=config.lr, eps=config.adam_epsilon)
    # 线性学习率调整机制,在warmup_step之前 线性的增加，在warmup_step之后线性降低
    num_warmup_step = config.warmup_step
    num_train_step = config.train_epochs * len(train_dataloader) // config.batch_size

    def lr_lambda(current_step):
        if current_step < num_warmup_step:
            return float(current_step) / float(max(1, num_warmup_step))
        return max(0.0, float(num_train_step - current_step)) / float(max(1, num_train_step - num_warmup_step))

    linear_scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda, last_epoch=-1)

    # Train
    logger.info("***** Running training *****")
    logger.info(" Num examples = %d" % len(train_dataloader))
    logger.info(" Num epochs = %d" % config.train_epochs)
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    seed_everything()

    save_steps = config.save_step if config.save_step > 0 else len(
        train_dataloader) // config.batch_size  # 每当经过这个step数之后保存模型
    dev_steps = config.dev_step if config.dev_step > 0 else len(
        train_dataloader) // config.batch_size  # 每当经过这个step数之后评估模型,并将结果输出
    # tensorboard 初始化
    writer = SummaryWriter(log_dir="cner_train")

    for epoch in range(config.train_epochs):
        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.to(config.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2], "labels": batch[3]}
            output = model(**inputs)
            loss, logits = output
            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            linear_scheduler.step()  # 更新lr的调整方案
            model.zero_grad()
            tr_loss += loss.item()  # 计算总loss
            global_step += 1
            if not config.debug:
                writer.add_scalar(tag="loss curve", scalar_value=loss, global_step=global_step)
            # 评估模型
            if dev_steps > 0 and global_step % dev_steps == 0:
                # 单个GPU的情况下,评估模型结果
                metric = SeqEntityScore(id2label, config.markup)
                for step, batch in enumerate(eval_dataloader):
                    model.eval()
                    batch = tuple(t.to(config.device) for t in batch)
                    with torch.no_grad():
                        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2],
                                  "labels": batch[3]}
                        loss, logits = model(**inputs)  # (batch_size,seq_length,hidden_size)
                        pred_labels = torch.argmax(logits, dim=-1)  # (32,118)
                        assert pred_labels.shape == batch[3].shape
                        metric.update(batch[3].cpu().numpy().tolist(), pred_labels.cpu().numpy().tolist())
                res, _ = metric.result()
                res = ",".join(f"{k}:{v}" for k, v in res.items())
                logger_mes = "epoch={},step={},loss={},".format(epoch, step, loss, res)
                logger.info(logger_mes)

            # 保存模型
            output_dir = config.output_dir
            if save_steps > 0 and global_step % save_steps == 0:
                # save model checkpoint
                output_dir = os.path.join(output_dir, "checkpoint-{}".format(global_step))  # 针对一个特定的global_step创建文件夹
                if not os.path.exists(output_dir):  # 创建文件夹
                    os.makedirs(output_dir)
                model_to_save = model.module if hasattr(model, "module") else model
                # transformers 提供的模型保存方式,采用这种方式，保存的文件默认为 pytorch_model.bin,config.json vocab.txt
                # 采用这种保存方式可以使用from_pretrained加载模型
                model_to_save.save_pretrained(output_dir)  # 保存训练的参数
                tokenizer.save_vocabulary(output_dir)  # 保存tokenizer的词汇表
                # 保存训练设置和超参数
                torch.save(config, os.path.join(output_dir, "training.bin"))
                logger.info("Saving model checkpoint to %s" % output_dir)
                # torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(linear_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                logger.info("Saving optimizer and scheduler states to %s" % output_dir)
        logger.info("\n")
        if "cuda" in str(config.device):
            torch.cuda.empty_cache()
        return global_step, tr_loss / global_step


def evaluate(config, model, tokenizer, evaluate_mes):
    processor = processors[config.task_name]()
    id2label = {i: label for i, label in enumerate(processor.get_labels())}
    label2id = {label: i for i, label in enumerate(processor.get_labels())}
    pred_output_dir = config.output_dir
    if not os.path.exists(pred_output_dir):
        os.makedirs(pred_output_dir)
    eval_dataset = load_and_cache_example(config, tokenizer, data_type="dev")
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=config.batch_size,
                                 collate_fn=collate_fn)
    # Eval
    logger.info("********* Running evaluation ********")
    logger.info(" Num examples = %d" % len(eval_dataset))
    logger.info("Batch size = %d" % config.batch_size)
    eval_loss = 0.0
    metric = SeqEntityScore(id2label, config.markup)
    for step, batch in enumerate(eval_dataloader):
        model.eval()
        batch = tuple(t.to(config.device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2], "labels": batch[3]}
            loss, logits = model(**inputs)  # (batch_size,seq_length,hidden_size)
            pred_labels = torch.argmax(logits, dim=-1)  # (32,118)
            assert pred_labels.shape == batch[3].shape
            metric.update(batch[3].cpu().numpy().tolist(), pred_labels.cpu().numpy().tolist())
    res, _ = metric.result()
    logger_mes = evaluate_mes + "acc:{},recall:{},f1:{}".format(res["acc"], res["recall"], res["f1"])
    logger.info(logger_mes)


def load_and_cache_example(config: CnerConfig, tokenizer, data_type="train"):
    processor = processors[config.task_name]()
    # logger.info("Create features from dataset file at %s" % config.data_dir)
    cached_features_file = os.path.join(config.data_dir, "cached_soft-{}-{}-{}-{}".format(
        data_type,
        str(config.model_type),
        str(config.train_max_seq_length if data_type == "train" else config.eval_max_seq_length),
        str(config.task_name),
    ))
    if os.path.exists(cached_features_file) and not config.overwrite_cache:
        logger.info("Loading features from cached files %s" % cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Create features from dataset file at %s" % config.data_dir)
        label_list = processor.get_labels()
        if data_type == "train":
            examples = processor.get_train_examples(config.data_dir)
        elif data_type == "dev":
            examples = processor.get_dev_examples(config.data_dir)
        else:
            examples = processor.get_test_examples(config.data_dir)
        features = convert_examples_to_features(examples=examples,
                                                label_list=label_list,
                                                tokenizer=tokenizer,
                                                max_seq_length=config.train_max_seq_length if data_type == "train"
                                                else config.eval_max_seq_length,
                                                cls_token=tokenizer.cls_token,
                                                cls_token_segment_id=0,
                                                sep_token=tokenizer.sep_token,
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=0)
        # save features
        logger.info("Saving features into cache file %s", cached_features_file)
        torch.save(features, cached_features_file)
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
    all_labels_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_lens, all_labels_ids)
    return dataset


def main():
    config = CnerConfig()
    # set output
    config.output_dir = config.output_dir + "{}".format(config.model_type)
    if not os.path.exists(config.output_dir): os.makedirs(config.output_dir)
    if os.path.exists(config.output_dir) and os.listdir(
            config.output_dir) and config.do_train and not config.overwrite_out_dir:
        raise ValueError("输出文件夹({})已经存在文件,请更改设置config.overwrite_out_dir".format(config.output_dir))
    # set gpu(不考虑并行训练)
    config.device = torch.device("cuda:0" if torch.cuda.is_available() and config.use_cuda else "cpu")
    # set seeding
    seed_everything()
    # set model
    num_labels = len(processors[config.task_name]().get_labels())
    config_class, model_class, tokenizer_class = MODEL_CLASSES[config.model_type]
    model_config = config_class.from_pretrained(config.model_name_or_path, num_labels=num_labels)
    tokenizer = tokenizer_class.from_pretrained(config.model_name_or_path)
    model_config.loss_type = config.loss_type
    model = model_class.from_pretrained(config.model_name_or_path, config=model_config)
    model.to(config.device)
    logger.info("training/evaluation parameters {}".format(config.__dict__))
    # train
    if config.do_train:
        global_step, tr_loss = train(config, model, tokenizer)
        logger.info("global step =%s, average loss =%s" % (global_step, tr_loss))
    # evaluation
    if config.do_eval:
        tokenizer = tokenizer.from_pretrained(config.output_dir)
        checkpoints = []
        pass
    # predict
    if config.do_predict:
        pass


if __name__ == '__main__':
    main()
