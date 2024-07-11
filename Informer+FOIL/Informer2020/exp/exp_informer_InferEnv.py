from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from exp.exp_basic import Exp_Basic
from models.model_env_W import AddEnv_W
from utils.tools import EarlyStopping, adjust_learning_rateE
from utils.metrics import metric

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import os
import time

import warnings
warnings.filterwarnings('ignore')
folder_name="ENV"
class Exp_Informer_InferEnv(Exp_Basic):
    def __init__(self, args, org_model=None):
        self.org_model = org_model
        for param in self.org_model.parameters():
            param.requires_grad = False

        super(Exp_Informer_InferEnv, self).__init__(args)
    
        
    def _build_model(self):
        data_len=self._get_data_len('train')
        model = AddEnv_W(out_len=self.args.pred_len,
            device=self.device,
            d_model=self.args.d_model,
            c_out=self.args.c_out,
            env_num=self.args.env_num,
            data_len=int(data_len),
            original_model=self.org_model
        ).float()
        return model
    def _get_data_len(self, flag):
        args = self.args

        data_dict = {
            'ETTh1':Dataset_ETT_hour,
            'ETTh2':Dataset_ETT_hour,
            'ETTm1':Dataset_ETT_minute,
            'ETTm2':Dataset_ETT_minute,
            'WTH':Dataset_Custom,
            'ECL':Dataset_Custom,
            'Solar':Dataset_Custom,
            'custom':Dataset_Custom,
        }
        Data = data_dict[self.args.data]
        timeenc = 0 if args.embed!='timeF' else 1

        if flag == 'test':
            shuffle_flag = False; drop_last = False; batch_size = 32; freq=args.freq
        elif flag=='pred':
            shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
            Data = Dataset_Pred
        elif flag=='env':
            shuffle_flag = True; drop_last = False; batch_size = args.batch_size; freq=args.freq
        else:
            shuffle_flag = True; drop_last = False; batch_size = args.batch_size; freq=args.freq
        if flag=='env':
            flag='train'
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )
        #print("GET DATA LEN",len(data_set))
        return len(data_set)
    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'ETTh1':Dataset_ETT_hour,
            'ETTh2':Dataset_ETT_hour,
            'ETTm1':Dataset_ETT_minute,
            'ETTm2':Dataset_ETT_minute,
            'WTH':Dataset_Custom,
            'ECL':Dataset_Custom,
            'Solar':Dataset_Custom,
            'custom':Dataset_Custom,
        }
        Data = data_dict[self.args.data]
        timeenc = 0 if args.embed!='timeF' else 1

        if flag == 'test':
            shuffle_flag = False; drop_last = False; batch_size = args.batch_size; freq=args.freq
        elif flag=='pred':
            shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
            Data = Dataset_Pred
        
        elif flag=='env':
            shuffle_flag = True; drop_last = False; batch_size = args.batch_size; freq=args.freq
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq=args.freq
        if flag=='env':
            flag='train'
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        parameters_to_optimize = [p for p in self.model.parameters() if p.requires_grad]

        model_optim = optim.Adam(parameters_to_optimize, lr=self.args.learning_rate)

        return model_optim
    
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        return criterion


    
    
    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        #
        path = os.path.join(self.args.checkpoints, setting)
        path = os.path.join(path, folder_name)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss_inv = []
            train_loss_var = []
            train_loss_org = []
            train_loss_sparse = []
            step=epoch
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark,indices) in enumerate(train_loader):
                iter_count += 1
                indices=indices.to(self.device)
                model_optim.zero_grad()
                pred_var, true, pred_inv,env_batch = self._process_one_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark,epoch,flag='train',indices=indices)
                inv_mean = pred_inv.mean(1, keepdim=True).detach()
                inv_std=pred_inv.std(1, keepdim=True).detach()
                var_mean = pred_var.mean(1, keepdim=True).detach()
                var_std=pred_var.std(1, keepdim=True).detach()
                true_mean = true.mean(1, keepdim=True).detach()
                true_std=true.std(1, keepdim=True).detach()
                pred_inv = (pred_inv - inv_mean)/inv_std
                pred_var = (pred_var - var_mean)/var_std
                true = (true - true_mean)/true_std
                loss_inv = criterion(pred_inv, true)
                train_loss_inv.append(loss_inv.item())
                loss_var= criterion(pred_var, true)
                train_loss_var.append(loss_var.item())
                #org_loss=self.model.soft_orthogonal_loss()
                #train_loss_org.append(org_loss.item())
                loss_re=self.reweight_loss(pred_var, pred_inv, true)
                loss=loss_re#+self.args.org*org_loss
                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss_inv: {2:.3f}| loss_var: {3:.3f}".format(i + 1, epoch + 1, loss_inv.item(),loss_var.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss_inv = np.average(train_loss_inv)
            train_loss_var = np.average(train_loss_var)
           
            print("Epoch: {0}, Steps: {1} | Train Inv : {2:.3f}  Train Var : {3:.3f}".format(
                epoch + 1, train_steps, train_loss_inv, train_loss_var))
            early_stopping(train_loss_var, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rateE(model_optim, int(epoch+1), self.args)
        

        
        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        self.collect_envs(train_data,train_loader)
        env_softmax= self.model.env_final
        
        return self.model,env_softmax

    def collect_envs(self,train_data,train_loader):
        self.model.eval()
        assigned_indices_tensor = torch.zeros(len(train_data), dtype=torch.bool)    
        train_data, train_loader = self._get_data(flag = 'env')  
        with torch.no_grad():
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark,indices) in enumerate(train_loader):                
                _, _,_, env_batch = self._process_one_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark,i,flag='train',indices=indices)
                indices = indices.to(self.device)
                assigned_indices_tensor[indices.cpu()] = True
               
                assert self.model.env_final[indices].shape == env_batch.squeeze().shape, \
                "Mismatched shapes between env_final and env_batch: {} vs {}".format
                self.model.env_final[indices] = env_batch.squeeze()
        assigned_indices = assigned_indices_tensor.tolist()
        unassigned_indices = [idx for idx, assigned in enumerate(assigned_indices) if not assigned]
    
        
    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark,step,flag="test",indices=-1):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()
        #print("batch_y",batch_y.shape)
        #batch_y torch.Size([128, 72, 7])
        batch_x_mark = batch_x_mark.float().to(self.device)
        
        batch_y_mark = batch_y_mark.float().to(self.device)

        # decoder input
        if self.args.padding==0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        elif self.args.padding==1:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)

        batch_y = batch_y.to(self.device)

        # encoder - decoder
       
        y_var,y_inv,env_batch = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,batch_y,step,flag,indices)
        if self.args.inverse:
            env_ag_out = dataset_object.inverse_transform(env_ag_out)
        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:]
        if flag=="train":
            return y_var, batch_y,y_inv,env_batch
        else:
            return y_var, batch_y
    def reweight_loss(self,pred_var, pred_inv, true):
        loss_inv = ((pred_inv - true) ** 2).mean(dim=1)
        loss_var = ((pred_var - true) ** 2).mean(dim=1)
        
        higher_var_loss = loss_var > loss_inv
        
        weights = torch.ones_like(loss_var)
        weights[higher_var_loss] =self.args.reweight
        
        weighted_loss_var = loss_var * weights
        
        return weighted_loss_var.mean()
def Smooth_Labels(arr, neighborhood):
    result = np.copy(arr)
    num_samples, num_classes = arr.shape
    num_change=0
    num_more=0
    for i in range(num_samples):
        class_count = np.zeros(num_classes)
        
        start_index = max(0, i - neighborhood)
        end_index = min(num_samples, i + neighborhood + 1)
        
        for j in range(start_index, end_index):
            class_count += arr[j]
        class_count += 0.5*arr[i]
        max_count = np.max(class_count)
        max_classes = np.where(class_count == max_count)[0]
        
        if len(max_classes) == 1:
            new_label = np.zeros(num_classes)
            new_label[max_classes[0]] = 1
            result[i] = new_label
            if np.argmax(new_label)!=np.argmax(arr[i]):
                num_change+=1
        else:
            num_more+=1
            result[i] = arr[i]
    print("CHANGED RATIO {}",num_change/num_samples)
    return result