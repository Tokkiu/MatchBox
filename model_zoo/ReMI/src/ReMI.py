# =========================================================================
# Copyright (C) 2020-2023. The ReMI Authors. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================


from torch import nn
import torch
from matchbox.pytorch.models import BaseModel
from matchbox.pytorch.layers import EmbeddingDictLayer
import torch.nn.functional as F


class ReMI(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="ReMI",
                 gpu=-1, 
                 learning_rate=1e-3, 
                 embedding_initializer="lambda w: nn.init.normal_(w, std=1e-4)", 
                 embedding_dim=10, 
                 user_id_field="user_id",
                 item_id_field="item_id",
                 user_history_field="user_history",
                 num_negs=1,
                 net_dropout=0,
                 net_regularizer=None,
                 embedding_regularizer=None,
                 similarity_score="dot",
                 interest_num=4,
                 beta=10,
                 reg_ratio=10,
                 **kwargs):
        super(ReMI, self).__init__(feature_map,
                                      model_id=model_id, 
                                      gpu=gpu, 
                                      embedding_regularizer=embedding_regularizer,
                                      net_regularizer=net_regularizer,
                                      num_negs=num_negs,
                                      embedding_initializer=embedding_initializer,
                                      **kwargs)
        self.similarity_score = similarity_score
        self.embedding_dim = embedding_dim
        self.user_id_field = user_id_field
        self.item_id_field = item_id_field
        self.user_history_field = user_history_field
        self.embedding_layer = EmbeddingDictLayer(feature_map, embedding_dim)
        self.dropout = nn.Dropout(net_dropout)
        self.reg_ratio = reg_ratio
        self.max_len = self.feature_map.feature_specs[self.user_history_field]['max_len']
        self.comi_aggregation = ComiRecAggregator(embedding_dim, interest_num=interest_num, seq_len=self.max_len)

        kwargs['noise'] = None
        kwargs['nce_loss_type'] = 'sampled'
        kwargs['noise_ratio'] = self.num_negs
        kwargs['norm_term'] = 0
        kwargs['beta'] = beta
        kwargs['item_num'] = self.feature_map.feature_specs[self.user_history_field]['vocab_size']

        self.compile(lr=learning_rate,**kwargs)
            
    def forward(self, inputs):
        """
        Inputs: [user_dict, item_dict, label]
        """
        user_dict, item_dict, labels = inputs[0:3]
        label_ids = item_dict[self.item_id_field].view(labels.size(0), self.num_negs + 1)[:,0].to(self.device)
        label_emb_dict = self.embedding_layer({self.item_id_field: label_ids}, feature_source="item")
        label_emb = label_emb_dict[self.item_id_field]
        readout, atten = self.user_tower(user_dict, label_emb=label_emb)
        item_vecs = self.item_tower(item_dict)
        y_pred = torch.bmm(item_vecs.view(readout.size(0), self.num_negs + 1, -1),
                               readout.unsqueeze(-1)).squeeze(-1)
        # item_corpus = self.embedding_layer.embedding_layers[self.item_id_field].weight
        # loss = self.loss_fn(label_ids.unsqueeze(-1), readout, item_corpus)
        loss = self.get_total_loss(y_pred, labels)
        # print(loss, self.attention_reg_loss(atten))
        loss += self.reg_ratio * self.attention_reg_loss(atten)
        return_dict = {"loss": loss, "y_pred": y_pred}
        return return_dict

    def user_tower(self, inputs, label_emb=None):
        user_inputs = self.to_device(inputs)
        user_emb_dict = self.embedding_layer(user_inputs, feature_source="user")
        user_history_emb = user_emb_dict[self.user_history_field]
        mask = user_history_emb.sum(dim=-1) != 0
        user_vec, atten = self.comi_aggregation(user_history_emb, label_emb, mask)
        if self.similarity_score == "cosine":
            user_vec = F.normalize(user_vec)

        if not self.training:
            return user_vec
        return user_vec, atten

    def item_tower(self, inputs):
        item_inputs = self.to_device(inputs)
        item_vec_dict = self.embedding_layer(item_inputs, feature_source="item")
        item_vec = self.embedding_layer.dict2tensor(item_vec_dict)
        if self.similarity_score == "cosine":
            item_vec = F.normalize(item_vec)
        return item_vec

    def attention_reg_loss(self, attention):
        C_mean = torch.mean(attention, dim=2, keepdim=True)
        C_reg = (attention - C_mean)
        C_reg = torch.bmm(C_reg, C_reg.transpose(1, 2)) / self.embedding_dim
        dr = torch.diagonal(C_reg, dim1=-2, dim2=-1)
        n2 = torch.norm(dr, dim=(1)) ** 2
        return n2.sum()


class ComiRecAggregator(nn.Module):

    def __init__(self, hidden_size, interest_num=4, seq_len=50):
        super(ComiRecAggregator, self).__init__()
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.num_heads = interest_num
        self.interest_num = interest_num
        self.hard_readout = True
        self.linear1 = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4, bias=False),
            nn.Tanh()
        )
        self.linear2 = nn.Linear(self.hidden_size * 4, self.num_heads, bias=False)

    def forward(self, item_eb, label_eb, mask):
        # item_eb = self.embeddings(item_list)
        item_eb = item_eb * torch.reshape(mask, (-1, self.seq_len, 1))

        # shape=(batch_size, maxlen, embedding_dim)
        item_eb = torch.reshape(item_eb, (-1, self.seq_len, self.hidden_size))

        # shape=(batch_size, maxlen, hidden_size*4)
        item_hidden = self.linear1(item_eb)
        # shape=(batch_size, maxlen, num_heads)
        item_att_w = self.linear2(item_hidden)
        # shape=(batch_size, num_heads, maxlen)
        item_att_w = torch.transpose(item_att_w, 2, 1).contiguous()

        atten_mask = torch.unsqueeze(mask, dim=1).repeat(1, self.num_heads, 1)  # shape=(batch_size, num_heads, maxlen)
        paddings = torch.ones_like(atten_mask, dtype=torch.float) * (-2 ** 32 + 1)

        item_att_w = torch.where(torch.eq(atten_mask, 0), paddings, item_att_w)
        item_att_w = F.softmax(item_att_w, dim=-1)  # shape=(batch_size, num_heads, maxlen)

        # shape=(batch_size, num_heads, embedding_dim)
        user_eb = torch.matmul(item_att_w,  # shape=(batch_size, num_heads, maxlen)
                                    item_eb  # shape=(batch_size, maxlen, embedding_dim)
                                    )  # shape=(batch_size, num_heads, embedding_dim)

        if self.training:
            # shape=(batch_size, embedding_dim)
            user_eb, _ = self.read_out(user_eb, label_eb)

        return user_eb, item_att_w

    def read_out(self, user_eb, label_eb):
        atten = torch.matmul(user_eb,  # shape=(batch_size, interest_num, hidden_size)
                             torch.reshape(label_eb, (-1, self.hidden_size, 1))  # shape=(batch_size, hidden_size, 1)
                             )  # shape=(batch_size, interest_num, 1)

        atten = F.softmax(torch.pow(torch.reshape(atten, (-1, self.interest_num)), 1),
                          dim=-1)  # shape=(batch_size, interest_num)

        if self.hard_readout:
            readout = torch.reshape(user_eb, (-1, self.hidden_size))[
                (torch.argmax(atten, dim=-1) + torch.arange(label_eb.shape[0],
                                                            device=user_eb.device) * self.interest_num).long()]
        else:
            readout = torch.matmul(torch.reshape(atten, (label_eb.shape[0], 1, self.interest_num)),
                                   # shape=(batch_size, 1, interest_num)
                                   user_eb  # shape=(batch_size, interest_num, hidden_size)
                                   )  # shape=(batch_size, 1, hidden_size)
            readout = torch.reshape(readout, (label_eb.shape[0], self.hidden_size))  # shape=(batch_size, hidden_size)
        selection = torch.argmax(atten, dim=-1)
        return readout, selection

