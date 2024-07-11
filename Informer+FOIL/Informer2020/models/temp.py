import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, env_num, env_dim, d_model, c_out):
        super(Model, self).__init__()
        assert env_dim == d_model  # 确保env_dim和d_model相等
        
        self.embed_env = nn.Embedding(env_num, env_dim)
        self.ln = nn.Linear(d_model, c_out, bias=False)

    def forward(self, inputs):
        # 取出embedding的权重
        embed_weights = self.embed_env.weight
        
        # 取出linear层的权重
        ln_weights = self.ln.weight.squeeze()  # 去除多余的维度
        
        # 计算差值
        diffs = embed_weights - ln_weights
        
        # 计算soft正交损失
        # diffs @ diffs^T计算了所有差向量的点积，我们希望除了对角线（自己与自己的点积）之外，其他都接近0
        soft_ortho_loss_matrix = diffs @ diffs.T
        soft_ortho_loss = soft_ortho_loss_matrix.pow(2).sum() - soft_ortho_loss_matrix.diag().pow(2).sum()
        
        return soft_ortho_loss

# 示例使用
env_num = 10
env_dim = 100
d_model = 100
c_out = 1

model = Model(env_num, env_dim, d_model, c_out)

# 假设我们有一些输入数据
inputs = torch.randint(0, env_num, (5,))  # 假设的输入indices

# 计算损失
loss = model(inputs)

print(loss)
