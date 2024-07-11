"""
mse:0.9502077102661133, mae:0.8153799176216125,msev:1.0486623048782349,mse_w:2.2096402645111084,mae_w:1.4386777877807617
"""


import torch
import numpy as np
import random
import pandas as pd
def set_seed(seed=1998):
    # Python
    random.seed(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

set_seed(3407)
                                    
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

if not 'Informer2020' in sys.path:
    sys.path += ['Informer2020']
stage1=1
if stage1==1:
    stage2=0
else:
    stage2=1
from utils.tools import dotdict
from exp.exp_informer_pre import Exp_Informer_Pre
from exp.exp_informer_InferEnv import Exp_Informer_InferEnv,Smooth_Labels
from exp.exp_informer_final import Exp_Informer_Final
args = dotdict()
args.model = 'informer' # model of experiment, options: [informer, informerstack, informerlight(TBD)]

args.data = 'custom' # data
args.root_path = '../dataset/exchange_rate/' # root path of data file
args.data_path = 'exchange_rate.csv' # data file
args.features = 'MS' # forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate
args.target = 'OT' # target feature in S or MS task
args.freq = 'w' # freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h
args.checkpoints = './informer_checkpoints' # location of model checkpoints

args.seq_len = 96 # input sequence length of Informer encoder
args.label_len = 48 # start token length of Informer decoder
args.pred_len = 24 # prediction sequence length

args.enc_in = 8 # encoder input size
args.dec_in = 8 # decoder input size
args.c_out = 1 # output size
args.factor = 3 # probsparse attn factor
args.d_model = 512 # dimension of model
args.n_heads = 8 # num of heads
args.e_layers = 2 # num of encoder layers
args.d_layers = 1 # num of decoder layers
args.d_ff = 512 # dimension of fcn in model
args.dropout = 0.05 # dropout
args.attn = 'prob' # attention used in encoder, options:[prob, full]
args.embed = 'timeF' # time features encoding, options:[timeF, fixed, learned]
args.activation = 'gelu' # activation
args.distil = True # whether to use distilling in encoder
args.output_attention = False # whether to output attention in ecoder
args.mix = True
args.padding = 0
args.freq = 'w'
args.learning_rate = 1e-2
args.loss = 'mse'
args.use_amp = False # whether to use automatic mixed precision training
args.num_workers = 0
args.itr = 1
args.des = 'exp'
args.use_gpu = True if torch.cuda.is_available() else False
args.gpu =0
args.use_multi_gpu = False
args.devices = '0'


args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False


if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ','')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

data_parser = {
    'ETTh1':{'data':'ETTh1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTh2':{'data':'ETTh2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTm1':{'data':'ETTm1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTm2':{'data':'ETTm2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'custom':{'data':'national_illness.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'custom':{'data':'exchange_rate.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[8,8,1]}
}
if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data']
    args.target = data_info['T']
    args.enc_in, args.dec_in, args.c_out = data_info[args.features]



args.detail_freq = args.freq
args.freq = args.freq[-1:]
print('Args in experiment:')
print(args)
devices = torch.device(f"cuda:{args.devices}")  # 使用 f-string 格式化

#第一轮的ERM训练模型
Exp_pre = Exp_Informer_Pre

args.lradj='type2'
args.learning_rate=1e-4
args.batch_size = 64
args.lr_adjust= {
            1: 1e-3, 2: 1e-4, 3: 1e-4, 8: 1e-4, 
            10: 1e-8, 15: 5e-10, 20: 5e-10
        }
args.train_epochs =5
args.patience = 3

args.env_num=4

if stage1==1:
    for ii in range(1):
        setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(args.model, args.data, args.features,
                    args.seq_len, args.label_len, args.pred_len,
                    args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.factor, args.embed, args.distil, args.mix, args.des, ii,args.sigma,args.alph,args.lam,args.hard_sum)
        exp_pre = Exp_pre(args)
        model_pre,best_model_path=exp_pre.train(setting)
        model_pre=exp_pre.model
        exp_pre.test(setting)
        torch.cuda.empty_cache()

    for i in range(1):
        Exp_env_infer = Exp_Informer_InferEnv
        #增加一层推断环境
        args.train_epochs =5
        args.patience =3
        args.batch_size = 128
        args.learning_rate = 1e-3
        args.lr_adjust= {
                -1:1e-5,1: 1e-3, 2: 1e-4, 3: 1e-4, 8: 1e-4, 
                10: 1e-4, 15: 5e-4, 20: 5e-4
            }
        args.reweight=0.2 #weights[higher_var_loss] =self.args.reweight

        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(args.model, args.data, args.features,
                        args.seq_len, args.label_len, args.pred_len,
                        args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.factor, args.embed, args.distil, args.mix, args.des, ii,args.sigma,args.alph,args.lam,args.hard_sum)
            # set experiments
            exp_env = Exp_env_infer(args,model_pre)
            # train
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            model_var,env_softmax=exp_env.train(setting)
            


        model_pre.to('cpu')
        env_softmax_np = env_softmax.cpu().detach().numpy()  # 首先确保它是在CPU上并转换为numpy
        env_softmax_2=Smooth_Labels(env_softmax_np,neighborhood=1)
        print(env_softmax_2.shape)
        softmax_filename = '1-EXE_NEW_'+str(args.env_num)+'class.csv'  # 根据迭代次数命名CSV文件
        pd.DataFrame(env_softmax_2).to_csv(softmax_filename, index=False)  # 使用pandas保存为CSV文件
        print(f'Saved env_softmax2 for iteration {ii} as {softmax_filename}')
        probabilities = env_softmax
        class_soft_scores = torch.sum(probabilities, dim=0)
        class_soft_proportions = class_soft_scores / probabilities.size(0)
        for i, soft_score in enumerate(class_soft_proportions):
            print(f"Class {i}: {soft_score.item():.4f} (Total soft score: {class_soft_scores[i].item():.4f})")
        torch.cuda.empty_cache()
if stage2==1:
    Exp_final = Exp_Informer_Final
    w_env_path='1-EXE_NEW_'+str(args.env_num)+'class.csv' 
    env_softmax = torch.from_numpy(pd.read_csv(w_env_path).values).to(devices).float()
    best_model_path='NA'
    flag="not load"

    args.cosin_lr=0
    args.learning_rate=1e-4
    args.batch_size =128
    # args.lr_adjust= {
    #             -1:1e-4,1: 1e-3, 2: 1e-4, 3: 1e-4, 8: 1e-4, 
    #             10: 1e-8, 15: 5e-10, 20: 5e-10
    #         }
    # args.lr_adjust= {
    #             -1:5e-5,1: 1e-4, 2: 1e-4,3:5e-4,4:5e-4,6:5e-5,
    #             10: 1e-8, 11:1e-9, 12:1e-10, 13:1e-11, 14:1e-12, 15:1e-13,
    #             15: 5e-10, 20: 5e-10
    #         }
    args.lr_adjust= {
                -1:5e-5,1: 1e-4, 2: 1e-4, 7:1e-8,8:1e-8,9:1e-8,
                10: 1e-8, 11:1e-9, 12:1e-10, 13:1e-11, 14:1e-12, 15:1e-13,

            }
    args.train_epochs=10
    args.patience = 10

    #第二轮的ERM训练模型
    args.start_epoch=5
    args.avg1=0.9
    args.var1=0.1
    args.avg2=0.8
    args.var2=0.2

    #最少环境的样本数量
    args.less_num = 3.0
    #scale_weight=sigma
    args.sigma=0.05
    #sbias_weight=alpa
    args.alpha=0.0
    args.swa_start =5
    #是不是使用余弦学习率调整
    
    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(args.model, args.data, args.features,
                    args.seq_len, args.label_len, args.pred_len,
                    args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.factor, args.embed, args.distil, args.mix, args.des, ii,args.sigma,args.alph)

        # set experiments
        exp_final = Exp_final(args,best_model_path,env_softmax,flag=flag)
        # train
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        model_pre,best_model_path=exp_final.train(setting)
        # test
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp_final.test(setting)
        torch.cuda.empty_cache()
