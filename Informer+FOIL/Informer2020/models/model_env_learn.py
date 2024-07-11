
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
    def __init__(self,  out_len=24,device=torch.device('cuda:0'),env_dim=512,env_num=6,data_len=-1,class_dim=[512,32,6],original_model=None,d_model=512,c_out=1,temper=1.0,HE_MLP=False,Normal_Env=False,Normal_std=-1):
        super(AddEnv, self).__init__()
        self.pred_len = out_len
        data_len=int(data_len)
        self.device = device
        self.d_model=d_model
        self.c_out=c_out
        self.env_num=env_num
        self.env_final = torch.zeros(data_len, int(env_num), device=self.device)
        self.original_model=original_model
        
        
        self.env_infer = self.make_mlp_softmax(class_dim,temper)
        self.embed_env = nn.Embedding(env_num, env_dim)
        
        assert class_dim[0] == d_model, "第一个维度应该是emb的维度"
        assert class_dim[-1] == env_num, "最后一个维度应该等于环境数量"
        self._init_weight(env_dim,HE_MLP,Normal_Env,Normal_std)
    def _init_weight(self,env_dim,HE_MLP,Normal_Env,Normal_std):
        if Normal_Env:
            nn.init.normal_(self.embed_env.weight, mean=0, std=Normal_std)
        else:
            nn.init.uniform_(self.embed_env.weight, -1 / math.sqrt(env_dim), 1 / math.sqrt(env_dim))
        self.embed_env.weight.requires_grad = True
        if HE_MLP:
            for module in self.env_infer.modules():
                if isinstance(module, nn.Linear):
                    # He initialization works well with ReLU activations, so we'll use it here
                    nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                    # If the Linear layer includes a bias term, initialize it to 0
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
        
    def renew(self):
        self.embed_env = nn.Embedding(env_num, emb_dim)
        self.env_w=nn.Embedding(data_len,env_num)
        self.var_predict=self.make_mlp(class_dim)
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, batch_y,step,scale,flag="test",indices=1.5,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):        
        self.original_model.eval()
        
        y_inv,inv_emb=self.original_model(x_enc, x_mark_enc, x_dec, x_mark_dec, batch_y,step,scale,flag="tune",indices=indices,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None)

        inv_emb_detach = inv_emb.detach()
        

        batch_size, seq_len, feat_dim = inv_emb_detach.size()
        inv_emb_pooled = torch.mean(inv_emb_detach, dim=1)
    
        env_w_normalized = self.env_infer(inv_emb_pooled)
        assert torch.allclose(torch.sum(env_w_normalized, dim=1), torch.ones_like(torch.sum(env_w_normalized, dim=1))), "Not all rows sum up to 1"
        env_emb=torch.mm(env_w_normalized, self.embed_env.weight)
        env_emb=expand_tensor(env_emb,int(self.pred_len))        
        elementwise_product = env_emb * inv_emb_detach
        summed_vectors = elementwise_product.sum(dim=1, keepdim=True)
        y_var=summed_vectors+y_inv
        
        if flag=="train":
            return y_var,y_inv,env_w_normalized
        else:
            return y_var,0
    def soft_orthogonal_loss(self):
        W_diff = self.embed_env.weight

        W_norm = F.normalize(W_diff, p=2, dim=1)

      
        I = torch.eye(W_norm.size(0), device=W_norm.device)

    
        W_transpose_W = torch.matmul(W_norm,W_norm.transpose(0, 1))
        assert W_transpose_W.size(0) == W_transpose_W.size(1)==W_norm.size(0), "The matrix W_norm is not square."
        assert torch.allclose(torch.diag(W_transpose_W), torch.ones_like(torch.diag(W_transpose_W))), \
            "The diagonal elements are not all close to 1."

        
        loss = torch.norm(W_transpose_W - I, p='fro') ** 2
        num_elements = W_norm.size(0) *(W_norm.size(0)-1) 
        loss_mean = loss / num_elements
        return loss_mean
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
    def soft_orthogonal_loss_OLD(self):
        """
        之前的orgloss有问题，计算的时样本概率的正交矩阵，而不是权重的正交矩阵 
        """
        W = F.normalize(self.env_w.weight, p=2, dim=1)  # 对权重按行进行L2范数归一化
        I = torch.eye(self.env_num, device=W.device)  # 创建单位矩阵
        W_transpose_W = torch.matmul(W.transpose(0, 1), W)  # 计算W^T * W
        loss = torch.norm(W_transpose_W - I, p='fro')**2  # 计算软正交化损失
        return loss
    def collect_envs(self, train_loader):
        self.eval() 
        all_envs = []

        with torch.no_grad():  
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, indices) in enumerate(train_loader):
                _, _, env_batch = self._process_one_batch(
                    batch_x, batch_x_mark, batch_y_mark, batch_y, indices=indices)
                all_envs.append(env_batch)

        print("save envs")
        self.env_final = torch.cat(all_envs, dim=0)
        print("save envs done",self.env_final.shape)
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
