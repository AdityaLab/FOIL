
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
import torch.nn.init as init
from RevIN import RevIN

def expand_tensor(env_emb, d):
    """Expand a tensor of shape (n, m) to (n, d, m) where every d values are the same."""
    env_emb_expanded = env_emb.unsqueeze(1)
    env_emb_repeated = env_emb_expanded.repeat(1, d, 1)
    return env_emb_repeated

class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu', 
                output_attention = False, distil=True, mix=True,
                device=torch.device('cuda:0'),penalty_anneal_iters=5,sigma=-1.0,alpha=-1.0,lam=0.1,hard_sum = 1.0,hidden_dim=1024,emb_dim=512,env_num=6,data_len=-1,class_dim=[512,32,6],rev_alpha=1.5,scale_weight=0.1):
        super(Informer, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        data_len=int(data_len)
        self.alpha=alpha
        self.output_attention = output_attention
        self.penalty_anneal_iters=penalty_anneal_iters
        self.device = device
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(1, d_model, embed, freq, dropout)
        Attn = ProbAttention if attn=='prob' else FullAttention
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )   
        self.d_model=d_model
        self.c_out=c_out
        self.ln=nn.Linear(d_model, c_out, bias=False)
        self.use_RevIN  = 1
        if self.use_RevIN==1:
            self.revin = RevIN(enc_in)
            self.revin_dec = RevIN(1)
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, batch_y, step, flag="test", indices=1.5,
            enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        x_dec=x_dec[:,:,-1].unsqueeze(-1)
        if self.use_RevIN==1:
            x_enc = self.revin(x_enc, 'norm')
            x_dec = self.revin_dec(x_dec, 'norm')
        #x_enc = torch.cat([x_enc[:,:,:-1] * self.scale+self.bias, x_enc[:,:,-1:]], dim=2)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out_OT = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.ln(dec_out_OT)
        if self.use_RevIN==1:
            dec_out = self.revin(dec_out, 'denorm')
        dec_out = dec_out[:,-self.pred_len:,-1].unsqueeze(-1)
        
        y_inv = dec_out
        
        if flag == "train":
            return y_inv
        elif flag == "tune":
            return y_inv, dec_out_OT
        else:
            return y_inv