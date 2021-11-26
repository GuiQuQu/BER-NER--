"""
    数据预处理工具类
"""


class DataProcessor(object):

    def get_train_examples(self, data_dir):
        """获取训练集"""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """获取验证集"""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        raise NotImplementedError()

    def get_labels(self):
        """获取标签"""
        raise NotImplementedError()

    @classmethod
    def _read_text(self, input_file):
        """读取txt文件,"""
        lines = []
        with open(input_file, "r", encoding="utf-8") as f:
            words, labels = [], []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if words:
                        lines.append({"words": words, "labels": labels})
                        words, labels = [], []
                else:
                    splits = line.split(" ")
                    words.append(splits[0])
                    if len(splits) > 1:
                        labels.append(splits[-1].replace("\n", ""))
                    else:
                        "数据文件中没有标签,直接写O"
                        labels.append("O")
            if words:
                lines.append({"words": words, "label": labels})
        return lines

    @classmethod
    def _read_json(self, input_files):
        """读取json文件，获取数据集"""
        pass


def get_entity_bios(seq, id2label):
    """将BIOS命名实体标志转变为命名实体
    Example
        >>> seq =['B-PER','I-PER','O','S-LOC']
        >>> get_entity_bios(seq)
        [['PER',0,1],['LOC',3,3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for index, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("S-"):  # S single
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = index
            chunk[2] = index
            chunk[0] = tag.split("-")[-1]
            chunks.append(chunk)
            chunk = [-1, -1, -1]
        elif tag.startswith("B-"):  # B begin
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = index
            chunk[0] = tag.split("-")[-1]
        elif tag.startswith("I-"):  # I intermediate
            _type = tag.split("-")[1]
            if _type == chunk[0]:
                chunk[2] = index
            if index == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks


def get_entity_bio(seq, id2label):
    """将BIO命名实体标志转变为命名实体
    Example
        >>> seq =['B-PER','I-PER','O','B-LOC']
        >>> get_entity_bios(seq)
        [['PER',0,1],['LOC',3,3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for index, tag in enumerate(seq):
        chunk = [-1, -1, -1]
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[0] = tag.split("-")[-1]
            chunk[1] = index
            chunk[2] = index
            if index == len(seq) - 1:
                chunks.append(chunk)
        elif tag.startswith("I-"):
            _type = tag.split("-")[-1]
            if _type == chunk[0]:
                chunk[2] = index
            else:
                raise ValueError("I标签应该和前面的B标签命名实体一致")
            if index == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
        return chunks


def get_entities(seq, id2label, markup="bios"):
    assert markup in ["bio", "bios"]
    if markup == "bio":
        return get_entity_bio(seq, id2label)
    else:
        return get_entity_bios(seq, id2label)
