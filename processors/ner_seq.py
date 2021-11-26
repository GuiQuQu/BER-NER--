import json

import torch
import logging
import os
import copy

from processors.utils_ner import DataProcessor

# __name__ 是python程序自带的一个属性，如果直接运行,__name__ ='__main__'
# 如果导入其他python程序中,__name__ = 该python文件名,即模块名
logger = logging.getLogger(__name__)


class InputExample(object):
    """单独的训练实例或者训练实例类型"""

    def __init__(self, guid, text_a, labels):
        """
        Constructs a InputExample
        Args:
            guid: Unique id for the example
            text_a: list.the words of sequence
            labels:(Optional) list.The labels for each word of the sequence
        """
        self.guid = guid
        self.text_a = text_a
        self.labels = labels

    def __repr__(self):
        """当打印class时，会调用这个函数，类似于java的toString"""
        return str(self.to_json_string())

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """single example feature set class for bert"""

    def __init__(self, input_ids, input_mask, input_len, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.input_len = input_len  # 不算padding得长度

    def __repr__(self):
        """当打印class时，会调用这个函数，类似于java的toString"""
        return str(self.to_json_string())

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer,
                                 cls_token_at_end=False, cls_token="[CLS]", cls_token_segment_id=1,
                                 sep_token="[SEP]", pad_on_left=False, pad_token=0, pad_token_segment_id=0,
                                 sequence_a_segment_id=0, mask_padding_with_zero=True, ):
    """Loads a data files into a list of 'InputBatch's
        'cls_token_at_end':是否将cls放在seq尾部
        - False(Default,BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
        - True(XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        'cls_token_segment_id' define the segment id associated to the CLS token (0 for BERT,2 for XLNet)
    """
    label_map = {label: i for i, label in enumerate(label_list)}  # 标签和label_id的映射对应关系
    features = []
    for ex_index, example in enumerate(examples):
        # logger 记录信息
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        if isinstance(example.text_a, list):
            example.text_a = " ".join(example.text_a)
        # 利用tokenizer,string -> token list,还不知道为什么tokenizer为什么按空格划分
        tokens = tokenizer.tokenize(example.text_a)
        label_ids = [label_map[x] for x in example.labels]
        # 对于过长的输入进行截断
        special_tokens_count = 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[:(max_seq_length - special_tokens_count)]
            label_ids = label_ids[:(max_seq_length - special_tokens_count)]

        # 看上去BERT的tokenizer.tokenize并不会加特殊的token,需要手动加入
        tokens += [sep_token]
        label_ids += [label_map["O"]]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [label_map["O"]]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            label_ids = [label_map["O"]] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids
        # 转换为BERT等预训练模型的输入
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        input_len = len(label_ids)
        # Zero-pad up to the sequence length
        padding_len = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_len) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_len) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_len) + segment_ids
            label_ids = ([pad_token] * padding_len) + label_ids
        else:
            input_ids += [pad_token] * padding_len
            input_mask += [0 if mask_padding_with_zero else 1] * padding_len
            segment_ids += [pad_token_segment_id] * padding_len
            label_ids += [pad_token] * padding_len

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        if ex_index < 5:
            logger.info("***Example***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
        features.append(
            InputFeatures(input_ids=input_ids, input_mask=input_mask, input_len=input_len, segment_ids=segment_ids,
                          label_ids=label_ids))
    return features


class CnerProcessor(DataProcessor):
    """Processor for the chinese ner data set"""

    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_text(os.path.join(data_dir, "train.char.bmes")), set_type="train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_text(os.path.join(data_dir, "dev.char.bmes")), set_type="dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_text(os.path.join(data_dir, "test.char.bmes")), set_type="test")

    def get_labels(self):
        return ["X", 'B-CONT', 'B-EDU', 'B-LOC', 'B-NAME', 'B-ORG', 'B-PRO', 'B-RACE', 'B-TITLE',
                'I-CONT', 'I-EDU', 'I-LOC', 'I-NAME', 'I-ORG', 'I-PRO', 'I-RACE', 'I-TITLE',
                'O', 'S-NAME', 'S-ORG', 'S-RACE', "[START]", "[END]"]

    def _create_examples(self, lines, set_type):
        examples = []
        # lines element {'words':List[(str)],'labels':List[(str)]}
        for i, line in enumerate(lines):
            if i == 0: continue
            guid = "%s-%s" % (set_type, i)
            text_a = line["words"]
            labels = []
            for x in line['labels']:
                if "M-" in x:
                    labels.append(x.replace("M-", "I-"))
                elif 'E-' in x:
                    labels.append(x.replace('E-', 'I-'))
                else:
                    labels.append(x)
            examples.append(InputExample(guid=guid, text_a=text_a, labels=labels))
        return examples


def collate_fn(batch):
    """
    batch (sequence,target,length) tuples
    """
    all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels = map(torch.stack, zip(*batch))
    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_attention_mask = all_attention_mask[:, :max_len]
    all_token_type_ids = all_token_type_ids[:, :max_len]
    all_labels = all_labels[:, :max_len]
    return all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_lens


processors = {"cner": CnerProcessor}
if __name__ == '__main__':
    # 假设是时间步T1的输出
    T1 = torch.tensor([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]])
    # 假设是时间步T2的输出
    T2 = torch.tensor([[10, 20, 30],
                       [40, 50, 60],
                       [70, 80, 90]])
    print(T1.size())
    print(T1.size(0))
    print(torch.stack((T1, T2), dim=0))
    print("---------------------")
    print(torch.stack((T1, T2), dim=1))
    print("----------------------")
    print(torch.stack((T1, T2), dim=2))
