import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertPreTrainedModel
from torch.nn import CrossEntropyLoss


class BertSoftmaxForNer(BertPreTrainedModel):
    def __init__(self, config):
        super(BertSoftmaxForNer, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_type = config.loss_type
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs.last_hidden_state  # (batch_size,seq_len,hidden_size)
        sequence_output = self.dropout(sequence_output)  # (batch_size,seq_len,hidden_size)
        logits = self.classifier(sequence_output)  # (batch_size,seq_len,num_labels) # 预测结果
        if labels is not None:  # 训练过程
            assert self.loss_type in ["ce"]
            loss_fct = CrossEntropyLoss(ignore_index=0)
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1  # 找出所有seq中有词的位置(去掉padding)
                active_logits = logits.view(-1, self.num_labels)[active_loss]  # 得到有词的位置的logtis
                active_labels = labels.view(-1)[active_loss]  # 获取有词位置的真实标签
                loss = loss_fct(active_logits, active_labels)  # 计算loss
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        else:
            return logits


if __name__ == '__main__':
    pass
