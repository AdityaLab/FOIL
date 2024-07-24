from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from exp.exp_basic import Exp_Basic
from models.model import Informer, InformerStack

from utils.tools import EarlyStopping, adjust_learning_rateF
from utils.metrics import metric
from torch import autograd
from torch.optim.swa_utils import AveragedModel, SWALR
import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import os
import time
from torch.optim.lr_scheduler import CosineAnnealingLR,ExponentialLR,LinearLR
import warnings
warnings.filterwarnings('ignore')
folder_name="FINAL"
class Exp_Informer_Final(Exp_Basic):
    def __init__(self, args,pre_model_path=None,env_softmax=None,flag="not load"):
        self.pre_model_path=pre_model_path
        assert self.pre_model_path is not None, "pre_model_path is not set!"
        self.env_softmax=env_softmax
        self.flag=flag
        
        assert self.env_softmax is not None, "env_softmax is not set!"
        super(Exp_Informer_Final, self).__init__(args)
    
    def _build_model(self):
        model_dict = {
            'informer':Informer,
            'informerstack':InformerStack,
        }
        
        data_len=self._get_data_len('train')
        if self.args.model=='informer' or self.args.model=='informerstack':
            e_layers = self.args.e_layers if self.args.model=='informer' else self.args.s_layers
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in, 
                self.args.c_out, 
                self.args.seq_len, 
                self.args.label_len,
                self.args.pred_len, 
                self.args.factor,
                self.args.d_model, 
                self.args.n_heads, 
                e_layers, # self.args.e_layers,
                self.args.d_layers, 
                self.args.d_ff,
                self.args.dropout, 
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                self.device,
                self.args.penalty_anneal_iters,
                self.args.sigma,
                self.args.alpha
            ).float()
            
            
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        if self.flag=="load":
            print("Loading pretrained model from")
            print("#####################################")
            model.load_state_dict(torch.load(self.pre_model_path))
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

        if flag == 'test' or flag == 'valid':
            shuffle_flag = False; drop_last = False; batch_size = 32; freq=args.freq
        elif flag=='pred':
            shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
            Data = Dataset_Pred
        
        else:
            shuffle_flag = True; drop_last = False; batch_size = args.batch_size; freq=args.freq
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

        if flag == 'test'or flag=='val':
            shuffle_flag = False; drop_last =False; batch_size =args.batch_size; freq=args.freq
        elif flag=='pred':
            shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
            Data = Dataset_Pred
        else:
            shuffle_flag = True; drop_last = False; batch_size = args.batch_size; freq=args.freq
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
        model_optim  = optim.RMSprop(self.model.parameters(),lr=self.args.learning_rate, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)

        return model_optim
    
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion,epoch,flag):
        self.model.eval()
        maeloss = nn.L1Loss()
        total_loss = []
        mae_loss = []
        self.model.eval() 
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark,indices) in enumerate(vali_loader):
            pred, true = self._process_one_batch(
                vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark,epoch,flag)
            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
            mae_loss.append(maeloss(pred, true).item())
        total_loss = np.average(total_loss)
        self.model.train()
        mae_loss = np.average(mae_loss)
        
        return total_loss,mae_loss
    
    
    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')
        path = os.path.join(self.args.checkpoints, setting)
        path = os.path.join(path, folder_name)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        swa_optim = AveragedModel(self.model)
        scheduler = CosineAnnealingLR(model_optim, T_max=self.args.Cosine_Step, eta_min=self.args.Cosine_Min)  # T_max和eta_min根据需要调整
        scheduler = ExponentialLR(model_optim, gamma=self.args.gamma)
        scheduler= LinearLR(model_optim, start_factor=1.0, total_iters=self.args.LinearStep, end_factor=1e-4)
        criterion =  self._select_criterion()
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        vali_loss,xxx = self.vali(vali_data, vali_loader,criterion,0,'test')
        test_loss,test_mae = self.vali(test_data, test_loader, criterion,0,'test')
        print("Epoch: {0} | Vali Loss: {1:.3f} |Test Loss: {2:.3f} Test MAE:{3:.3f}".format(
               0,vali_loss, test_loss,test_mae))
        for epoch in range(self.args.train_epochs):
            if epoch==13:
                print("skip epoch 13")
                continue
            iter_count = 0
            train_loss_avg = []
            train_loss_var = []
            step=epoch
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark,indices) in enumerate(train_loader):
                iter_count += 1
                indices=indices.to(self.device)
                with torch.no_grad():
                    batch_env=self.env_softmax[indices]
                model_optim.zero_grad()

                pred_inv, true = self._process_one_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark,epoch,flag='train',indices=indices)
                # Check for NaNs/Infs in predictions and true values
                if torch.isnan(pred_inv).any() or torch.isinf(pred_inv).any():
                    print("NaN or Inf found in predictions, skipping batch")
                    continue  # Skip this batch
                if torch.isnan(true).any() or torch.isinf(true).any():
                    print("NaN or Inf found in true values, skipping batch")
                    continue  # Skip this batch
                epsilon = 1e-18
                squared_error = (pred_inv - true) ** 2
                if torch.isnan(squared_error).any() or torch.isinf(squared_error).any():
                    print("NaN or Inf found in squared error, skipping batch")
                    continue  # Skip this batch
                log_squared_error = torch.log(squared_error + 1 + epsilon)
                if torch.isnan(log_squared_error).any() or torch.isinf(log_squared_error).any():
                    print("NaN or Inf found in log_squared_error, skipping batch")
                    continue  # Skip this batch
                
                loss_all = torch.mean(log_squared_error, dim=1, keepdim=False)
                loss_avg = torch.mean(loss_all)
                inv_mean = pred_inv.mean(1, keepdim=True).detach()
                pred_shape = pred_inv - inv_mean
                true_mean = true.mean(1, keepdim=True).detach()
                true_shape = true - true_mean
                shape_error= torch.mean((pred_shape - true_shape) ** 2, dim=1, keepdim=False)
                loss_var=loss_var_call(shape_error, batch_env, min_samples=3)
                train_loss_avg.append(loss_avg.item())
                train_loss_var.append(loss_var.item())
                
                if epoch<=self.args.start_epoch:
                    
                    loss=self.args.avg1*loss_avg+self.args.var1*loss_var
                else:
                    loss=self.args.avg2*loss_avg+self.args.var2*loss_var
               
                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss_avg: {2:.3f} | loss_var: {3:.3f}".format(i + 1, epoch + 1, loss_avg.item(),loss_var.item()))
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
                
            if epoch > self.args.swa_start:
                swa_optim.update_parameters(self.model)
            train_loss_avg = np.average(train_loss_avg)
            train_loss_var = np.average(train_loss_var)
          
            vali_loss,xx = self.vali(vali_data, vali_loader,criterion,epoch,'val')
            test_loss,test_mae= self.vali(test_data, test_loader, criterion,epoch,'test')

            print("Epoch: {0}, Steps: {1} | Train Avg : {2:.3f} | Train Var : {3:.10f} | Vali Loss: {4:.3f} |Test Loss: {5:.3f} Test MAE: {6:.3f} ".format(
                epoch + 1, train_steps, train_loss_avg,train_loss_var, vali_loss, test_loss,test_mae))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            print(self.args.cosin_lr)
            if self.args.cosin_lr==0:
                adjust_learning_rateF(model_optim, int(epoch+1), self.args)
            if self.args.cosin_lr==1:
                current_lr = model_optim.param_groups[0]['lr']
                print("current_lr",current_lr)
                scheduler.step()
            
        
        

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        
        
        return self.model,best_model_path

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        
        self.model.eval()
        
        preds = []
        trues = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark,indices) in enumerate(test_loader):
            pred, true= self._process_one_batch(
                test_data, batch_x, batch_y, batch_x_mark, batch_y_mark,0,'test')
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae,mse,rmse,mape,mspe,mse_var,mse_w,mae_w = metric(preds, trues,0.1)
        print('mse:{}, mae:{},msev:{},mse_w:{},mae_w:{}'.format(mse, mae,mse_var,mse_w,mae_w))

        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path+'pred.npy', preds)
        np.save(folder_path+'true.npy', trues)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')
        
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            path = os.path.join(path, folder_name)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()
        
        preds = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(pred_loader):
            pred, true = self._process_one_batch(
                pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        
        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        np.save(folder_path+'real_prediction.npy', preds)
        
        return

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark,step,flag="test",indices=-1):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()
        
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
       
        y_inv= self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,batch_y,step,flag,indices)
        if self.args.inverse:
            env_ag_out = dataset_object.inverse_transform(env_ag_out)
        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:]
       
        return y_inv, batch_y

 
def loss_var_call(shape_error, batch_env, min_samples):
    
    env_sample_counts = batch_env.sum(dim=0)

    valid_envs = env_sample_counts >= min_samples

    valid_loss = []
    for i, is_valid in enumerate(valid_envs):
        if is_valid:
            env_loss = shape_error[batch_env[:, i] == 1]
            valid_loss.append(env_loss)

    loss_variance = torch.var(torch.cat(valid_loss)) if valid_loss else torch.tensor(0.0)

    return loss_variance
