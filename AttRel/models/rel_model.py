from torch import nn
from transformers import *
import torch
import json
from torch.nn.utils.rnn import pad_sequence
class AttModel(nn.Module):
    def __init__(self, config):
        super(AttModel, self).__init__()
        self.config = config
        self.bert_dim = config.bert_dim
        self.bert_encoder = BertModel.from_pretrained("bert-base-cased", cache_dir='./pre_trained_bert')
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        self.rel_embedding = nn.Linear(self.config.rel_num, self.bert_dim)
        self.relation_matrix = nn.Linear(self.bert_dim * 3, self.config.rel_num * self.config.tag_size)
        self.projection_matrix = nn.Linear(self.bert_dim * 2, self.bert_dim * 3)

        self.dropout = nn.Dropout(self.config.dropout_prob)
        self.dropout_2 = nn.Dropout(self.config.entity_pair_dropout)
        self.activation = nn.ReLU()

        #self.multihead_attention = nn.MultiheadAttention(embed_dim=self.bert_dim, num_heads=self.config.num_heads,dropout=0.2)

    def get_encoded_text(self, token_ids, mask):
        # [batch_size, seq_len, bert_dim(768)]
        encoded_text = self.bert_encoder(token_ids, attention_mask=mask)[0]
        #print("😊",encoded_text.shape)
        #print(encoded_text.shape)
        with open("data/NYT/rel.json", 'r') as f:
            rel = json.load(f)
        #print(rel)
        #rel_types = list(rel[0].values())
        #rel_types.sort(key=len)
        #rel_types_encoded = [self.bert_tokenizer.encode(r, add_special_tokens=False, ) for r in rel_types]
        encoded_input = self.bert_tokenizer(rel, padding=True, truncation=True, return_tensors='pt')

        encoded_input = encoded_input.to(encoded_text.device)
        rel_types_output = self.bert_encoder(**encoded_input)
        rel_types_encoded = rel_types_output.last_hidden_state
        #print(rel_types_encoded.shape)
        encoded_text_rel =torch.randn(encoded_text.size(0),encoded_text.size(1),encoded_text.size(2)).to(encoded_text.device)
        for i in range(encoded_text.size(0)):
            matrix_b = encoded_text[i]
            #print("=====================\n",encoded_text[i].size())
            #print(i,":",matrix_b)
            for j in range(rel_types_encoded.size(0)):
                matrix_a = rel_types_encoded[i]
                #print(j, ":", matrix_a)
                query = matrix_a
                key = matrix_b
                scores = torch.matmul(query, key.t())
                scaled_scores = scores / (key.size(-1) ** 0.5)
                weights = torch.softmax(scaled_scores, dim=-1)

                # 计算加权和
                matrix_c = torch.matmul(weights.t(), matrix_a)

                #print(matrix_c.size())  # 输出：torch.Size([112, 768])
                #encoded_text[i]=encoded_text[i]+matrix_c
                #encoded_text[i]=torch.add(encoded_text[i],matrix_c)
                matrix_b=matrix_b + matrix_c
            encoded_text_rel[i]=matrix_b

        return encoded_text_rel

    def triple_score_matrix(self, encoded_text, train=True):
        # encoded_text: [batch_size, seq_len, bert_dim(768)] 1,2,3
        batch_size, seq_len, bert_dim = encoded_text.size()
        # head: [batch_size, seq_len * seq_len, bert_dim(768)] 1,1,1, 2,2,2, 3,3,3
        head_representation = encoded_text.unsqueeze(2).expand(batch_size, seq_len, seq_len, bert_dim).reshape(
            batch_size, seq_len * seq_len, bert_dim)
        # tail: [batch_size, seq_len * seq_len, bert_dim(768)] 1,2,3, 1,2,3, 1,2,3
        tail_representation = encoded_text.repeat(1, seq_len, 1)
        # [batch_size, seq_len * seq_len, bert_dim(768)*2]
        entity_pairs = torch.cat([head_representation, tail_representation], dim=-1)

        # [batch_size, seq_len * seq_len, bert_dim(768)*3]
        entity_pairs = self.projection_matrix(entity_pairs)

        entity_pairs = self.dropout_2(entity_pairs)

        entity_pairs = self.activation(entity_pairs)

        # [batch_size, seq_len * seq_len, rel_num * tag_size] -> [batch_size, seq_len, seq_len, rel_num, tag_size]
        triple_scores = self.relation_matrix(entity_pairs).reshape(batch_size, seq_len, seq_len, self.config.rel_num,
                                                                   self.config.tag_size)

        if train:
            # [batch_size, tag_size, rel_num, seq_len, seq_len]
            return triple_scores.permute(0, 4, 3, 1, 2)
        else:
            # [batch_size, seq_len, seq_len, rel_num]
            return triple_scores.argmax(dim=-1).permute(0, 3, 1, 2)

    def forward(self, data, train=True):
        # [batch_size, seq_len]
        token_ids = data['token_ids']
        # [batch_size, seq_len]
        mask = data['mask']
        # [batch_size, seq_len, bert_dim(768)]
        encoded_text = self.get_encoded_text(token_ids, mask)
        encoded_text = self.dropout(encoded_text)
        # [batch_size, rel_num, seq_len, seq_len]
        output = self.triple_score_matrix(encoded_text, train)

        return output