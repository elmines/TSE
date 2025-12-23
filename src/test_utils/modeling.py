import torch.nn as nn
from transformers import BertModel
from transformers import AutoModel


# BERTweet or BERT
class bert_classifier(nn.Module):

    def __init__(self, config, plm_model):
        
        super(bert_classifier, self).__init__()
        self.dropout = nn.Dropout(0.)
        self.relu = nn.ReLU()
        
        if plm_model == 'bertweet':
            self.bert = AutoModel.from_pretrained("vinai/bertweet-base", local_files_only=False)
        elif plm_model == 'bert':
            self.bert = BertModel.from_pretrained("bert-base-uncased", local_files_only=False)
        self.bert.pooler = None
        self.linear_main = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.out_main = nn.Linear(self.bert.config.hidden_size, int(config['num_labels']))
        self.linear_aux = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.out_aux = nn.Linear(self.bert.config.hidden_size, int(config['num_tar']))
        
    def forward(self, x_input_ids, x_seg_ids, x_atten_masks, x_len, task_id):

        last_hidden = self.bert(input_ids=x_input_ids, attention_mask=x_atten_masks, token_type_ids=x_seg_ids)
        cls = last_hidden[0][:,0]
        query = self.dropout(cls)
        
        if task_id[0] == 0:
            linear = self.relu(self.linear_main(query))
            out = self.out_main(linear)
        else:
            linear = self.relu(self.linear_aux(query))
            out = self.out_aux(linear)
        
        return out

    