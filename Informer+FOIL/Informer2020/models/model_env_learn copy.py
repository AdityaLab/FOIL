
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import optim, nn, autograd
from utils.masking import TriangularCausalMask, ProbMask
from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from models.decoder import Decoder, DecoderLayer
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import DataEmbedding
import math

class AddEnv(nn.Module):
    def __init__(self,  out_len=24,device=torch.device('cuda:0'),env_dim=512,env_num=6,data_len=-1,class_dim=[512,32,6],original_model=None,d_model=512,c_out=1):
        super(AddEnv, self).__init__()
        self.pred_len = out_len
        data_len=int(data_len)
        self.device = device
        self.d_model=d_model
        self.c_out=c_out
        self.embed_env = nn.Embedding(env_num, env_dim)
        self.env_w=nn.Embedding(data_len,env_num)
        assert class_dim[0] == d_model+env_dim, "第一个维度应该等于之和"
        assert class_dim[-1] == c_out, "最后一个维度应该等于 c_out"
        self.var_predict=self.make_mlp(class_dim)
        self.original_model=original_model
    def get_env_softmax(self):
        env_w=self.env_w
        env_w_normalized=F.softmax(env_w, dim=1)
        return env_w_normalized
    def make_mlp(self, dims,flag="not last relu"):
        layers = []
        if flag=="last relu":
            tt=1
        else:
            tt=2
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - tt:  # 如果不是最后一层，添加ReLU激活
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)
    def renew(self):
        self.embed_env = nn.Embedding(env_num, emb_dim)
        self.env_w=nn.Embedding(data_len,env_num)
        self.var_predict=self.make_mlp(class_dim)
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, batch_y,step,scale,flag="test",indices=1.5,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):        
        self.original_model.eval()
        y_inv,inv_emb=self.original_model(x_enc, x_mark_enc, x_dec, x_mark_dec, batch_y,step,scale,flag="tune",indices=indices,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None)
        env_w=self.env_w(indices)
        env_w_normalized=F.softmax(env_w, dim=1)
        env_emb=torch.mm(env_w_normalized, self.embed_env.weight)
        env_emb=expand_tensor(env_emb,int(self.pred_len))
        assert inv_emb.shape[0] == env_emb.shape[0], \
        f"Shape mismatch! inv_emb: {inv_emb.shape}, env_emb: {env_emb.shape}"
        assert inv_emb.shape[1] == env_emb.shape[1], \
        f"Shape mismatch! inv_emb: {inv_emb.shape}, env_emb: {env_emb.shape}"
    
        y_var=self.var_predict(torch.cat([inv_emb,env_emb],dim=2))
        
        if flag=="train":
            return y_var,y_inv
        else:
            return y_var,0


def expand_tensor(env_emb, d):
    """Expand a tensor of shape (n, m) to (n, d, m) where every d values are the same."""
    env_emb_expanded = env_emb.unsqueeze(1)
    env_emb_repeated = env_emb_expanded.repeat(1, d, 1)
    return env_emb_repeated






















class InformerStack(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, additional_emb=512, n_heads=8, e_layers=[3,2,1], d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                output_attention = False, distil=True, mix=True,
                device=torch.device('cuda:0')):
        super(InformerStack, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention
        self.device = device
        print("self.device",self.device)
        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, additional_emb, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, additional_emb, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder

        inp_lens = list(range(len(e_layers))) # [0,1,2,...] you can customize here
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                    additional_emb, n_heads, mix=False),
                        additional_emb,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for l in range(el)
                ],
                [
                    ConvLayer(
                        additional_emb
                    ) for l in range(el-1)
                ] if distil else None,
                norm_layer=torch.nn.LayerNorm(additional_emb)
            ) for el in e_layers]
        self.encoder = EncoderStack(encoders, inp_lens)
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                additional_emb, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                additional_emb, n_heads, mix=False),
                    additional_emb,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(additional_emb)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=additional_emb, out_channels=c_out, kernel_size=1, bias=True)
        self.ln_add = nn.Linear(additional_emb, c_out, bias=True)
        self.ln_OT = nn.Linear(d_ff, c_out, bias=True)
        self.ln_con=nn.Linear(c_out, c_out, bias=True)
        for lin in [self.ln_add, self.ln_OT,self.ln_con]:
            nn.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:,-self.pred_len:,:], attns
        else:
            return dec_out[:,-self.pred_len:,:] # [B, L, D]
