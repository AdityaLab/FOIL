
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import optim, nn, autograd
import torch.nn.init as init
from utils.masking import TriangularCausalMask, ProbMask
from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from models.decoder import Decoder, DecoderLayer
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import DataEmbedding
import math

class AddEnv_W(nn.Module):
    def __init__(self,  out_len=24,device=torch.device('cuda:0'),
                 env_num=6,data_len=-1,original_model=None,d_model=512,c_out=1):
        super(AddEnv_W, self).__init__()
        self.pred_len = out_len
        data_len=int(data_len)
        self.device = device
        self.d_model=d_model
        self.c_out=c_out
        self.env_num=env_num
        self.env_final = torch.zeros(data_len, int(env_num), device=self.device)
        self.original_model=original_model
        self.embed_env_W = nn.Embedding(env_num, d_model)
        self._init_weight('xavier_uniform')
    def _init_weight(self,type):
        if type == 'normal':
            init.normal_(self.embed_env_W.weight, mean=0, std=0.1)
        elif type == 'uniform':
            init.uniform_(self.embed_env_W.weight, a=-0.1, b=0.1)
        elif type == 'xavier_normal':
            init.xavier_normal_(self.embed_env_W.weight)
        elif type == 'xavier_uniform':
            init.xavier_uniform_(self.embed_env_W.weight)
        elif type == 'kaiming_normal':
            init.kaiming_normal_(self.embed_env_W.weight, mode='fan_out', nonlinearity='relu')
        elif type == 'kaiming_uniform':
            init.kaiming_uniform_(self.embed_env_W.weight, mode='fan_out', nonlinearity='relu')
        else:
            raise ValueError("Unknown weight initialization method: {}".format(type))
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, batch_y,step,flag="test",indices=1.5,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):        
        self.original_model.eval()
        y_inv,inv_emb=self.original_model(x_enc, x_mark_enc, x_dec, x_mark_dec, batch_y,step,flag="tune",indices=indices,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None)
        inv_emb_detach = inv_emb.detach()
        with torch.no_grad():
            y_true = batch_y[:,-self.pred_len:,-1:]
        y_vars = self.compute_y_var(inv_emb_detach,y_inv) 
        with torch.no_grad():
             best_env_indices_int,best_env_indices_one_hot = self.select_best_env(y_vars, y_true)
        y_var_best = torch.stack([y_vars[env_idx][i] for i, env_idx in enumerate(best_env_indices_int)])
        if flag=="train":
            return y_var_best,y_inv,best_env_indices_one_hot 
        else:
            return y_var_best,0
    def compute_y_var(self, inv_emb,y_inv):
        y_vars = []
        for i in range(self.env_num):
            env_weight = self.embed_env_W.weight[i]  
            y_var = torch.matmul(inv_emb, env_weight) 
            y_vars.append(y_var.unsqueeze(-1)[:,-self.pred_len:,:])  
        return y_vars

    def select_best_env(self, y_vars, y_true):
        batch_size = y_true.size(0)
        env_num = len(y_vars)
        y_vars_stack = torch.stack(y_vars)[:,:,-self.pred_len:,:]
        var_mean =y_vars_stack.mean(1, keepdim=True).detach()
        var_std=y_vars_stack.std(1, keepdim=True).detach()
        y_pred=(y_vars_stack-var_mean)/var_std
        y_true_0= y_true.unsqueeze(0)
        y_true_mean = y_true_0.mean(1, keepdim=True)
        y_true_std=y_true_0.std(1, keepdim=True)
        y_true_1=(y_true-y_true_mean)/y_true_std
        errors = torch.mean((y_pred - y_true_1) ** 2, dim=[2, 3])
        best_env_indices = torch.argmin(errors, dim=0)
        env_indices_one_hot = torch.zeros(batch_size, env_num, device=self.device)
        env_indices_one_hot[torch.arange(batch_size), best_env_indices] = 1

        return best_env_indices,env_indices_one_hot
    #暂时保留
    def soft_orthogonal_loss(self):
        W_diff = self.embed_env.weight

        # 归一化这些差异向量
        W_norm = F.normalize(W_diff, p=2, dim=1)

        # 创建一个单位矩阵，用于计算正交损失
        I = torch.eye(W_norm.size(0), device=W_norm.device)

        # 计算 W_norm * W_norm的转置
        W_transpose_W = torch.matmul(W_norm,W_norm.transpose(0, 1))
        # 确保归一化后的权重形状是一个方阵，即行数和列数相等
        assert W_transpose_W.size(0) == W_transpose_W.size(1)==W_norm.size(0), "The matrix W_norm is not square."
        # 确保对角线上的元素接近1
        assert torch.allclose(torch.diag(W_transpose_W), torch.ones_like(torch.diag(W_transpose_W))), \
            "The diagonal elements are not all close to 1."

        # 计算损失：Frobenius范数的平方
        loss = torch.norm(W_transpose_W - I, p='fro') ** 2
        num_elements = W_norm.size(0) *(W_norm.size(0)-1) # 因为是方阵，所以行数的平方即为元素总数
        loss_mean = loss / num_elements
        return loss_mean
    #暂时保留
    def make_mlp_softmax(self, dims, temperature=1.0):
        """
        Create a multilayer perceptron with softmax on the output layer.
        
        Parameters:
            dims (list): A list of layer dimensions.
            temperature (float): The temperature parameter for softmax.
        """
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i+1], bias=False))
            if i < len(dims) - 2:  # Add ReLU for all layers except last
                layers.append(nn.ReLU())
        
        # Replace the last ReLU with Softmax for classification
        # Apply temperature to the logits before softmax
        def softmax_with_temperature(x):
            return F.softmax(x / temperature, dim=1)
        
        layers.append(nn.Sequential(
            nn.Linear(dims[-2], dims[-1], bias=False),
            nn.Softmax(dim=1)  # Assuming that we are dealing with batch data
        ))
        
        return nn.Sequential(*layers)
    
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
