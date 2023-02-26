import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

import math
from transformers.models.bert.modeling_bert import BertConfig, BertOnlyMLMHead
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEmbeddings, BertEncoder, BertPooler

import numpy as np

import transformers


# w = torch.from_numpy(1.0 - np.load('./w.npy')).cuda()
# w = torch.from_numpy(np.ones(52)).cuda()
# w[0] = 0.05
# w[1] = 0.05

# w = np.array([1.00997009,1.00589623, 1.02881844, 1.03095975, 1.09090909, 1.05235602,
#  1.01162791, 1.00602047, 1.19230769, 1.0591716,  1.04385965, 1.52631579,
#  1.00345185, 1.01838235, 1.02061856 ,1.0057971 , 1.05434783 ,1.05235602,
#  1.14705882, 1.04048583 ,1.02008032 ,1.02309469 ,1.20408163, 1.18181818,
#  1.02832861, 1.35714286 ,1.03690037 ,1.08403361, 1.58823529, 1.0140056,
#  1.05524862, 1.00805802 ,1.10309278 ,1.03012048, 1.0097561 , 1.01251564,
#  1.04950495, 1.00865052 ,1.06451613 ,1.04854369, 1.05617978, 1.05952381,
#  1.5 ,       1.04016064 ,1.08064516, 1.01605136, 1.12345679, 1.01677852,
#  1.03597122, 1.01207729 ,1.06756757 ,1.12345679])
# w = torch.from_numpy(w).cuda()
# cirtion = nn.BCEWithLogitsLoss(weight=w, reduction="mean")

cirtion = nn.BCEWithLogitsLoss(reduction="mean")



# cirtion2 =nn.MSELoss()

class MultiModal(nn.Module):
    def __init__(self, bert_path):
        super().__init__()
        self.bert_text_encoder1 = BertModel.from_pretrained(bert_path)
        self.bert_text_encoder2 = BertModel.from_pretrained(bert_path)

        # self.bert_text_encoder = transformers.LongformerModel.from_pretrained(bert_path_long)
#         for p in self.parameters():                
#             p.requires_grad = False

        bert_output_size = 768
        self.trans = EncoderLayer()
        self.soft_att = softattention(bert_output_size)
        self.fccls = nn.Linear(bert_output_size, 52)
       
    def forward(self, title_input=None, title_mask=None, targets=None,  helpfu = None):
        features1 = self.bert_text_encoder1(title_input[:, :512], title_mask[:, :512])['last_hidden_state']
        features2 = self.bert_text_encoder2(title_input[:, 512:], title_mask[:, 512:])['last_hidden_state']
        features = torch.cat((features1, features2),dim=1) 
        features = self.trans(features)
        features = self.soft_att(features)
        out =  self.fccls(features) 
        
        if not self.training:
            return out
        else:
            loss = cirtion(out, targets)
            return loss

 
class softattention(nn.Module):
    def __init__(self, hidden_size):
        super(softattention, self).__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def get_attn(self, reps, mask=None):
        res = torch.unsqueeze(reps, 1)
        attn_scores = self.attn(reps).squeeze(2)
        if mask is not None:
            attn_scores = mask * attn_scores
        attn_weights = attn_scores.unsqueeze(2)
        attn_out = torch.sum(reps * attn_weights, dim=1)
        return attn_out

    def forward(self, reps, mask=None):
        attn_out = self.get_attn(reps, mask)
        return attn_out


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V):
        n_heads = 12
        d_model = 768
        d_k = 64
        d_v = 64
        d_ff = 768* 4
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_heads, seq_len, seq_len]
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        n_heads = 12
        d_model = 768
        d_k = 64
        d_v = 64
        d_ff = 768 * 4
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.dds = nn.Linear(n_heads * d_v, d_model)
        self.dlayy = nn.LayerNorm(d_model)
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.d_ff = d_ff

    def forward(self, Q, K, V):
        # q: [batch_size, seq_len, d_model], k: [batch_size, seq_len, d_model], v: [batch_size, seq_len, d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,
                                                                                 2)  # q_s: [batch_size, n_heads, seq_len, d_k]
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,
                                                                                 2)  # k_s: [batch_size, n_heads, seq_len, d_k]
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,
                                                                                 2)  # v_s: [batch_size, n_heads, seq_len, d_v]

        # context: [batch_size, n_heads, seq_len, d_v], attn: [batch_size, n_heads, seq_len, seq_len]
        context = ScaledDotProductAttention()(q_s, k_s, v_s)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1,
                                                            self.n_heads * self.d_v)  # context: [batch_size, seq_len, n_heads, d_v]
        output = self.dds(context)
        output = self.dlayy(output + residual)
        return output  # output: [batch_size, seq_len, d_model]


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        n_heads = 12
        d_model = 768
        d_k = 64
        d_v = 64
        d_ff = 768 * 4
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        return self.fc2(gelu(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        n_heads = 12
        d_model = 768
        d_k = 64
        d_v = 64
        d_ff = 768 * 4
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs):
        enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs)  # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, seq_len, d_model]
        return enc_outputs


def gelu(x):
    """
      Implementation of the gelu activation function.
      For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
      0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
      Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

