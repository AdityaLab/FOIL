
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

class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu', 
                output_attention = False, distil=True, mix=True,
                device=torch.device('cuda:0'),penalty_anneal_iters=5,sigma=1.0,alpha=0.5,lam=0.1,hard_sum = 1.0 ):
        super(Informer, self).__init__()
        self.pred_len = out_len
        self.attn = attn


        


        self.output_attention = output_attention
        self.penalty_anneal_iters=penalty_anneal_iters
        # EncDoc-m6
        self.device = device
        additional_emb=8
        self.enc_embedding = DataEmbedding(enc_in-1, additional_emb, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(1, additional_emb, embed, freq, dropout)

        # EncDoc-OT1:targrt 变量的预测结果
        self.enc_embedding_OT = DataEmbedding(1, d_model, embed, freq, dropout)
        self.dec_embedding_OT = DataEmbedding(1, d_model, embed, freq, dropout)

        #其余的变量
        
        #d_ff_ENV=8
       
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder
        

        self.encoder_OT = Encoder(
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
        self.decoder_OT = Decoder(
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
        """
        self.ln_add = nn.Linear(additional_emb, c_out*1, bias=True)
        self.ln_OT = nn.Linear(d_model, c_out*1, bias=True)
        print("d_ff",d_ff)
        self.ln_con=nn.Linear(c_out, c_out, bias=True)
        """
        #self.ln_A = nn.Linear(additional_emb+d_model, c_out*16, bias=True)
        #self.ln_B=nn.Linear(c_out*16, c_out*8, bias=True)
        #self.ln_C=nn.Linear(c_out*8, c_out*1, bias=True)
        self.ln=nn.Linear(d_model, c_out, bias=True)
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, batch_y,step,scale,flag,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None,selection_flag=0):
    
        x_enc_OT = x_enc[:, :, -1:]
        x_dec_OT = x_dec[:, :, -1:]
        
        enc_out_OT = self.enc_embedding_OT(x_enc_OT, x_mark_enc)
        enc_out_OT, attns = self.encoder_OT(enc_out_OT, attn_mask=enc_self_mask)
        #128 18(pre_len) 512(emb_size)
        #step1: 计算每个节点的环境表征、
        #step2: 根据环境表征算每个channel的权重，使用HRM中的gate机制或者dida中的1-w
        #step3: 分别走两个分支，一个variant 一个是invariant
        #step4: 将两个分支的结果相加，得到最终的预测结果
        dec_out_OT = self.dec_embedding_OT(x_dec_OT, x_mark_dec)
        dec_out_OT = self.decoder_OT(dec_out_OT, enc_out_OT, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        #print("dec_out_OT",dec_out_OT.shape) dec_out_OT torch.Size([128, 42, 512])
        
        #相当于输出是label len+pred len
        final_out=self.ln(dec_out_OT)
        
        if self.output_attention:
            return final_out[:,-self.pred_len:,:],attns,dec_out_OT
        else:
            return final_out[:,-self.pred_len:,:],dec_out_OT

class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.weight_init()

    def weight_init(self):
        torch.nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        return self.linear(x)

def pretty(vector):
    if type(vector) is list:
        vlist = vector
    elif type(vector) is np.ndarray:
        vlist = vector.reshape(-1).tolist()
    else:
        vlist = vector.view(-1).tolist()
    return "[" + ", ".join("{:+.4f}".format(vi) for vi in vlist) + "]"

# Feature selection part
class FeatureSelector(nn.Module):
    def __init__(self, input_dim, sigma):
        super(FeatureSelector, self).__init__()
        self.mu = torch.nn.Parameter(0.00 * torch.randn(input_dim, ), requires_grad=True)
        self.noise = torch.randn(self.mu.size())
        self.sigma = sigma
        self.input_dim = input_dim

    def renew(self):
        self.mu = torch.nn.Parameter(0.00 * torch.randn(self.input_dim, ), requires_grad=True)
        self.noise = torch.randn(self.mu.size())

    def forward(self, prev_x):
        z = self.mu + self.sigma * self.noise.normal_() * self.training
        stochastic_gate = self.hard_sigmoid(z)
        new_x = prev_x * stochastic_gate
        return new_x

    def hard_sigmoid(self, x):
        return torch.clamp(x + 0.5, 0.0, 1.0)

    def regularizer(self, x):
        return 0.5 * (1 + torch.erf(x / math.sqrt(2)))

    def _apply(self, fn):
        super(FeatureSelector, self)._apply(fn)
        self.noise = fn(self.noise)
        return self





























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
