from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import PatchTST, AquaCast
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop, visual_plot, visual_rain, visual_acc
from utils.metrics import metric
from pytorch_forecasting.metrics import QuantileLoss


import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

from dtaidistance import dtw

class Exp_Main_exo(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main_exo, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'PatchTST': PatchTST,
            'AquaCast': AquaCast,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        
        print('all parameter', sum(p.numel() for p in model.parameters()))
        print('trainable prm', sum(p.numel() for p in model.parameters() if p.requires_grad))
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    import torch

    def quantile_loss(
        z: torch.Tensor,
        z_hat: torch.Tensor,
        quantiles
    ) -> torch.Tensor:
        """
        Quantile (pinball) loss:
        - summed over quantiles
        - averaged over batch, forecast steps, variables

        Args:
            z:      target values, shape [B, T, V]
            z_hat:  quantile predictions, shape [B, T, V, Q]
            quantiles: list or 1D tensor of quantile levels (length Q)

        Returns:
            scalar loss
        """
        if not torch.is_tensor(quantiles):
            quantiles = torch.tensor(
                quantiles, device=z_hat.device, dtype=z_hat.dtype
            )
        else:
            quantiles = quantiles.to(device=z_hat.device, dtype=z_hat.dtype)

        # reshape for broadcasting: [1, 1, 1, Q]
        q = quantiles.view(1, 1, 1, -1)

        # expand target: [B, T, V, 1]
        z = z.unsqueeze(-1)

        # pinball loss components
        under = torch.relu(z - z_hat)
        over  = torch.relu(z_hat - z)

        # sum over quantiles
        loss = q * under + (1.0 - q) * over     # [B, T, V, Q]
        loss = loss.sum(dim=-1)                 # [B, T, V]

        # average over batch, time, variables
        return loss.mean()

    def _select_criterion(self):
        """
        Returns a callable `criterion(outputs, batch_y) -> scalar loss`
        using PyTorch-Forecasting QuantileLoss.

        Expected:
          outputs: [B, T, V, Q]
          batch_y: [B, T, V]
        """
        self.quantiles = getattr(self.args, "quantiles", [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])  # or wherever configs live
        qloss = QuantileLoss(quantiles=self.quantiles)

        def criterion(outputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            # outputs: [B, T, V, Q], target: [B, T, V]
            if outputs.ndim != 4:
                raise ValueError(f"Expected outputs [B,T,V,Q], got {tuple(outputs.shape)}")
            if target.ndim != 3:
                raise ValueError(f"Expected target  [B,T,V], got {tuple(target.shape)}")

            B, T, V = target.shape
            Q = outputs.shape[-1]
            if outputs.shape[:3] != (B, T, V):
                raise ValueError(
                    f"Shape mismatch: outputs {tuple(outputs.shape)} vs target {tuple(target.shape)}"
                )

            # Treat each variable as an extra batch item so averaging matches your spec
            # outputs: (B, T, V, Q) -> (B*V, T, Q)
            y_hat = outputs.permute(0, 2, 1, 3).reshape(B * V, T, Q)
            # target : (B, T, V) -> (B*V, T)
            y = target.permute(0, 2, 1).reshape(B * V, T)

            # QuantileLoss.loss returns unreduced loss; we average over everything
            loss_unreduced = qloss.loss(y_hat, y)
            return loss_unreduced.mean()

        return criterion
    
    def compute_dtw(self, outputs, batch_y):
        """
        Computes the DTW-based hit rate for each batch element and signal.

        Parameters:
        - outputs: numpy array of shape [batch, signal_length, signal_number] (predictions)
        - batch_y: numpy array of shape [batch, signal_length, signal_number] (ground-truth)

        Returns:
        - hit_rates: numpy array of shape [batch, signal_number], percentage of hits per batch-signal pair
        """
        batch_size, signal_length, num_signals = outputs.shape
        errors = np.zeros((batch_size, num_signals))  # Store hit rate per batch-signal

        for b in range(batch_size):
            for s in range(num_signals):
                # Get the time series for the current batch element and signal
                pred_signal = outputs[b, :, s]
                true_signal = batch_y[b, :, s]

                # Compute DTW warping path
                path = dtw.warping_path(pred_signal, true_signal)

                # Extract aligned pairs
                aligned_preds = np.array([pred_signal[p[0]] for p in path])
                aligned_truth = np.array([true_signal[p[1]] for p in path])

                # Compute error
                error = np.mean(np.abs(aligned_preds - aligned_truth))  # Mean Absolute Error
                # error = np.mean((aligned_preds - aligned_truth) ** 2)  # Mean Squared Error
                errors[b, s] = error  # Store result

        return errors

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                if self.args.exo_future:
                    batch_x, exo_future = batch_x
                    exo_future = exo_future.float().to(self.device)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if 'TST' in self.args.model or 'Aqua' in self.args.model:
                    if self.args.exo_future:
                        outputs = self.model(batch_x, exo_future)
                    else:
                        outputs = self.model(batch_x)
                else:
                    print('model is not implemented')
                    break
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:, :]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                # remove index -2 and -3 if args.exo_future and args.exo are True, but keep the index -1
                if self.args.features == 'M':
                    target = batch_y[:, :, -1:]
                    if self.args.exo:
                        batch_y = batch_y[:, :, :-2]
                        # concatenate the target to the batch_y
                        batch_y = torch.cat((batch_y, target), axis=-1)
                    else:
                        pass
                    assert batch_y.shape[-1] == outputs.shape[-2], f"batch_y shape: {batch_y.shape}, outputs shape: {outputs.shape}, shapes do not match"


                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
            
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)

        train_plot = []
        valid_plot = []
        test_plot = []
        
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                if self.args.exo_future:
                    batch_x, exo_future = batch_x
                    exo_future = exo_future.float().to(self.device)
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if 'TST' in self.args.model or 'Aqua' in self.args.model:
                    if self.args.exo_future:
                        outputs = self.model(batch_x, exo_future)
                    else:
                        outputs = self.model(batch_x)
                else:
                    print('model is not implemented')
                    break
                
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:, :]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                # remove index -2 and -3 if args.exo_future and args.exo are True, but keep the index -1
                if self.args.features == 'M':
                    target = batch_y[:, :, -1:]
                    if self.args.exo:
                        batch_y = batch_y[:, :, :-2]
                        # concatenate the target to the batch_y
                        batch_y = torch.cat((batch_y, target), axis=-1)
                    else:
                        pass
                    assert batch_y.shape[-1] == outputs.shape[-2], f"batch_y shape: {batch_y.shape}, outputs shape: {outputs.shape}, exo {self.args.exo}, exo_future {self.args.exo_future}, shapes do not match"
                
                # loss
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
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
                    
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            #save records of training losses
            train_plot.append(train_loss)
            valid_plot.append(vali_loss)
            test_plot.append(test_loss)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        # #train-valid loss plot
        # visual_plot(train_plot, valid_plot, test_plot, os.path.join(folder_path, 'training_plot_mse' + '.pdf'))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        #--- Add Sin Signal ----
        if self.args.data == 'custom_sin':
            sin_y = test_data.sin_y
        else:
            sin_y = 0

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location='cuda:0'))

        criterion = [self._select_criterion()]
        criterion.append(nn.L1Loss())
        
        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                if self.args.exo_future:
                    batch_x, exo_future = batch_x
                    exo_future = exo_future.float().to(self.device)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if 'TST' in self.args.model or 'Aqua' in self.args.model:
                    if self.args.exo_future:
                        outputs = self.model(batch_x, exo_future)
                    else:
                        outputs = self.model(batch_x)
                else:
                    print('model is not implemented')   
                    break

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:, :]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                # remove index -2 if args.exo_future and args.exo are True, but keep the index -1
                if self.args.features == 'M':
                    target = batch_y[:, :, -1:]
                    if self.args.exo:
                        batch_y = batch_y[:, :, :-2]
                        # concatenate the target to the batch_y
                        batch_y = torch.cat((batch_y, target), axis=-1)
                    else:
                        pass
                    assert batch_y.shape[-1] == outputs.shape[-2], f"batch_y shape: {batch_y.shape}, outputs shape: {outputs.shape}, shapes do not match"

                # mse_label = criterion[0](outputs[0, :, -1], batch_y[0, :, -1])
                # mae_label = criterion[1](outputs[0, :, -1], batch_y[0, :, -1])

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                
                # Remove Sine signal if it was added in data_loader
                outputs -= sin_y
                batch_y -= sin_y

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                # if i % 20 == 0:
                #     input = batch_x.detach().cpu().numpy()
                #     # for s in range(input.shape[-1]): #signal number
                #     for s in range(true.shape[-1]): #signal number
                #         gt = np.concatenate((input[0, :, s], true[0, :, s]), axis=0)
                #         pd = np.concatenate((input[0, :, s], pred[0, :, s]), axis=0)
                #         path = os.path.join(folder_path, f'{s}_mse{mse_label.item():.5f}_mae{mae_label.item():.5f}_{i}_0.pdf') #{s}_MSE_MAE_{i}_0:  signal, MSE, MAE, batch number, batch element # to have the each signals in order of good to bad samples
                #         visual(gt, pd, path)

        preds = np.array(preds)
        trues = np.array(trues)
        inputx = np.array(inputx)

        preds = preds.reshape(-1, preds.shape[-3], preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        mean, var = test_data.scaler.mean_, test_data.scaler.scale_
        preds_real = preds * var[-1] + mean[-1]
        trues_real = trues * var[-1] + mean[-1]

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr, R2, EVS, nse, mse_individual, mae_individual, rmse_individual, r2_individual, sql, sql_point = metric(preds, trues, self.quantiles)
        mae_real, mse_real, rmse_real, mape_real, mspe_real, rse_real, corr_real, R2_real, EVS_real, nse_real, mse_individual_real, mae_individual_real, rmse_individual_real, r2_individual_real, sql_real, sql_point_real = metric(preds_real, trues_real, self.quantiles)

        print('mse:{}, mae:{}, rmse:{}, R2:{}, nse:{}\n'.format(mse, mae, rmse, R2, nse))
        print('mse_real:{}, mae_real:{}, rmse_real:{}, R2_real:{}, nse_real:{}\n'.format(mse_real, mae_real, rmse_real, R2_real, nse_real))
        print('Individual:\n')
        print('mse:{}, mae:{}, rmse:{}, R2:{}\n'.format(mse_individual, mae_individual, rmse_individual, r2_individual))
        print('mse_real:{}, mae_real:{}, rmse_real:{}, R2_real:{}\n'.format(mse_individual_real, mae_individual_real, rmse_individual_real, r2_individual_real))
        print('SQL:{}, SQL_point:{}'.format(sql, sql_point))
        print('SQL_real:{}, SQL_point_real:{}'.format(sql_real, sql_point_real))

        f = open("result.txt", 'a')
        f.write('Test End Time: {}\n'.format(time.strftime("%Y.%m.%d,%H:%M:%S", time.localtime())))
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rmse:{}, R2:{}\n'.format(mse, mae, rmse, R2))
        f.write('mse_real:{}, mae_real:{}, rmse_real:{}, R2_real:{}\n'.format(mse_real, mae_real, rmse_real, R2_real))
        f.write('Individual:\n')
        f.write('mse:{}, mae:{}, rmse:{}, R2:{}\n'.format(mse_individual, mae_individual, rmse_individual, r2_individual))
        f.write('mse_real:{}, mae_real:{}, rmse_real:{}, R2_real:{}\n'.format(mse_individual_real, mae_individual_real, rmse_individual_real, r2_individual_real))
        f.write('SQL:{}, SQL_point:{}\n'.format(sql, sql_point))
        f.write('SQL_real:{}, SQL_point_real:{}\n'.format(sql_real, sql_point_real))
        # f.write('DTW Error, test all, MAE threshold: {}, close samples: {} {}%, far samples:{} {}%, all:{}'.format(mse_threshold, cls_c, cls_2all, far_c, far_2all, all_c))
        # f.write('\n')
        # f.write('mse individual:{}'.format(mse_individual))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'mse_individual.npy', mse_individual)
        # np.save(folder_path + 'mae_individual.npy', mae_individual)
        # np.save(folder_path + 'rmse_individual.npy', rmse_individual)
        # np.save(folder_path + 'r2_individual.npy', r2_individual)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
        return


    def test_all(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        mean, var = test_data.scaler.mean_, test_data.scaler.scale_
        # check if test_data has exo_forecast_scaler
        try:
            mean_exo, var_exo = test_data.exo_forecast_scaler.mean_, test_data.exo_forecast_scaler.scale_
        except AttributeError:
            mean_exo, var_exo = None, None
        min_, max_ = test_data.normal.data_min_, test_data.normal.data_max_
        inverse_transform = False
        print(self.args.exo_future, self.args.exo)
        # print('dataset scaler', test_data.scaler.mean_, test_data.scaler.var_)
        #--- Add Sin Signal ----
        if self.args.data == 'custom_sin':
            sin = np.concatenate((test_data.sin_x, test_data.sin_y), axis=0).reshape(-1)
            sin_y = test_data.sin_y
            sin_x = test_data.sin_x
        else:
            sin = 0
            sin_y = 0
            sin_x = 0

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location='cuda:0'))

        criterion = [self._select_criterion()]
        criterion.append(nn.L1Loss())

        mse_threshold = 0.107 #0.091 #0.45
        # hit_threshold = 0.9
        cls_c = [0]
        far_c = [0]
        
        preds = []
        trues = []
        rains = []
        inputx = []

        folder_path = './test_results/' + 'test_all_DTW_' + setting + '/'
        print(folder_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                if self.args.exo_future:
                    batch_x, exo_future = batch_x
                    exo_future = exo_future.float().to(self.device)
                print(i, batch_x.shape, batch_y.shape)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if 'TST' in self.args.model or 'Aqua' in self.args.model:
                    if self.args.exo_future:
                        outputs = self.model(batch_x, exo_future)
                    else:
                        outputs = self.model(batch_x)
                else:
                    print('model is not implemented')   
                    break

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                if self.args.features != 'S':
                    if self.args.exo_future and self.args.exo: # when both exo and exo_future are in dataset, the order of the features is [..., exogenous, exogenous_future, Target]
                        rain    = batch_x[:, -self.args.seq_len:, -2].to(self.device)
                        rain_y  = exo_future[...,-1].to(self.device)
                    elif (not self.args.exo_future) and self.args.exo: # when only exo is in dataset, the order of the features is [..., exogenous, Target]
                        rain    = batch_x[:, -self.args.seq_len:, -2].to(self.device)
                        rain_y  = None
                    else:
                        rain    = None
                        rain_y  = None
                    rain_index = -2
                else:
                    rain    = None
                    rain_y  = None
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                # batch_x = batch_x[:, -self.args.pred_len:, 0:]

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                inpt    = batch_x.detach().cpu().numpy()
                if self.args.features != 'S':
                    if rain is not None:
                        rain    = rain.detach().cpu().numpy()
                    if rain_y is not None:
                        rain_y  = rain_y.detach().cpu().numpy()
                # Remove Sine signal if it was added in data_loader
                outputs -= sin_y
                batch_y -= sin_y
                inpt    -= sin_x

                # remove index -2 and -3 if args.exo_future and args.exo are True, but keep the index -1
                if self.args.features == 'M':
                    target = batch_y[:, :, -1:]
                    if self.args.exo:
                        batch_y = batch_y[:, :, :-2]
                        # concatenate the target to the batch_y
                        batch_y = np.concatenate((batch_y, target), axis=-1)
                    else:
                        pass
                    assert batch_y.shape[-1] == outputs.shape[-2], f"batch_y shape: {batch_y.shape}, outputs shape: {outputs.shape}, shapes do not match"

                
                # Metrics
                # DTW-based hit rate
                # DTW_error_per_signal = self.compute_dtw(outputs, batch_y)
            
                # nice_signals = DTW_error_per_signal < mse_threshold
                
                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                
                # b = nice_signals.shape[0]
                # s = nice_signals.shape[1]
                # cls_c += nice_signals.sum(axis=0)
                # far_c += b - nice_signals.sum(axis=0)

                for b in [0]: #batch elements e.g. [0, 15, 48, 128] or range(inpt.shape[0])
                    for s in range(1): #signal number. or range(inpt.shape[-1]). for MS it should be only the target signal with index 0
                        # repeat inpt if quantile output: inpt [b, sl, nf] -> [b, sl, nf, q] to match the shape of pred and true for visualizing the input together with the quantile predictions and true values
                        if self.quantiles is not None:
                            past = np.repeat(inpt[:, :, -1:], repeats=len(self.quantiles), axis=-1)
                        else:
                            past = inpt[:, :, -1:]
                        
                        print(f'inpt shape: {inpt.shape}, pred shape: {pred.shape}, true shape: {true.shape}, past shape: {past.shape}')
                        gt = np.concatenate((inpt[b, :, -1], true[b, :, -1]), axis=0)
                        pd = np.concatenate((past[b, ...], pred[b, :, -1]), axis=0)
                        # mse = np.mean((pred[b, ..., -1] - true[b, ..., -1])**2) # only for the forecasted part
                        # print('mse: ', mse)
                        print(f'gt shape: {gt.shape}, pd shape: {pd.shape}')

                        if not(self.args.exo_future or self.args.exo):
                            rn = None
                        elif self.args.exo_future:
                            rn = np.concatenate((rain[b, :], rain_y[b, :]), axis=0)
                        else:
                            rn = rain[b, :]
                        print(f'rn shape: {rn.shape}' if rn is not None else 'rn is None')
                        print(f'rain shape: {rain[b, :].shape}, rain_y shape: {rain_y[b, :].shape}')
                        
                        if inverse_transform:
                            # reverse the standardization transform
                            gt = gt * var[-1] + mean[-1]
                            pd = pd * var[-1] + mean[-1]
                            if rn is not None:
                                if mean_exo is not None and var_exo is not None:
                                    rn[:len(rain[b, :])] = rn[:len(rain[b, :])] * var[rain_index] + mean[rain_index]
                                    rn[len(rain[b, :]):] = rn[len(rain[b, :]):] * var_exo[-1] + mean_exo[-1]
                                else:
                                     rn = rn * var[rain_index] + mean[rain_index]

                        path = os.path.join(folder_path, f'plot_quantile_{s}_{i}_{b}.pdf') # {i}_{b}_{s}: batch number, batch element, signal
                        # visual(gt, pd, path) 
                        # arr = np.array([gt, pd, rn], dtype=object)
                        # np.save(path.replace('.pdf', '.npy'), arr)
                        print(path)
                        visual_rain(gt, pd, rn, 'MSE', None, path, quantiles=self.quantiles) 

        # all_c = cls_c + far_c
        # print(all_c)
        # print(cls_c)
        # print(far_c)
        # cls_2all =  [a/b for a, b in zip(cls_c, all_c)]
        # far_2all =  [a/b for a, b in zip(far_c, all_c)]
        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()
        preds = np.array(preds)
        trues = np.array(trues)
        inputx = np.array(inputx)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr, R2, EVS, nse, mse_individual, mae_individual, rmse_individual, r2_individual, sql, sql_point = metric(preds, trues, self.quantiles)
        print('mse:{}, mae:{}, rmse:{}, R2:{}, nse:{}\n'.format(mse, mae, rmse, R2, nse))
        print('Individual:\n')
        print('mse:{}, mae:{}, rmse:{}, R2:{}\n'.format(mse_individual, mae_individual, rmse_individual, r2_individual))
        print('sql:{}, sql_point:{}'.format(sql, sql_point))
        
        f = open("result.txt", 'a')
        f.write('Test End Time: {}\n'.format(time.strftime("%Y.%m.%d,%H:%M:%S", time.localtime())))
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rmse:{}, R2:{}\n'.format(mse, mae, rmse, R2))
        f.write('Individual:\n')
        f.write('mse:{}, mae:{}, rmse:{}, R2:{}\n'.format(mse_individual, mae_individual, rmse_individual, r2_individual))
        f.write('DTW Error, test all, MAE threshold: {}, close samples: {} {}%, far samples:{} {}%, all:{}\n'.format(mse_threshold, cls_c, cls_2all, far_c, far_2all, all_c))
        f.write('sql:{}, sql_point:{}\n'.format(sql, sql_point))
        # f.write('\n')
        # f.write('mse individual:{}'.format(mse_individual))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'mse_individual.npy', mse_individual)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
        return

    def accuracy_threshold_plot(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        mean, var = test_data.scaler.mean_, test_data.scaler.var_
        min_, max_ = test_data.normal.data_min_, test_data.normal.data_max_
        inverse_transform = True
        # print('dataset scaler', test_data.scaler.mean_, test_data.scaler.var_)

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location='cuda:0'))

        criterion = [self._select_criterion()]
        criterion.append(nn.L1Loss())

        mse_threshold = np.arange(0.02, 1.0, 0.01).tolist()
        
        out_signal_n = self.args.enc_in - self.args.exo if self.args.features == 'M' else 1 

        cls_c = [[0] * out_signal_n] * len(mse_threshold) 
        far_c = [[0] * out_signal_n] * len(mse_threshold)
        
        preds = []
        trues = []
        rains = []
        inputx = []
        folder_path = './test_results/' + 'test_all_DTW_' + setting + '/'
        print(folder_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                if self.args.exo_future:
                    batch_x, exo_future = batch_x
                    exo_future = exo_future.float().to(self.device)
                print(batch_x.shape, batch_y.shape)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                if 'TST' in self.args.model or 'Aqua' in self.args.model:
                    if self.args.exo_future:
                        outputs = self.model(batch_x, exo_future)
                    else:
                        outputs = self.model(batch_x)
                else:
                    print('model is not implemented')   
                    break

                f_dim = -1 if self.args.features == 'MS' else 0

                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                # remove index -2 and -3 if args.exo_future and args.exo are True, but keep the index -1
                # print(self.args.exo_future, self.args.exo)
                if self.args.features == 'M':
                    target = batch_y[:, :, -1:]
                    if self.args.exo:
                        batch_y = batch_y[:, :, :-2]
                        # concatenate the target to the batch_y
                        batch_y = torch.cat((batch_y, target), axis=-1)
                    else:
                        pass
                    assert batch_y.shape[-1] == outputs.shape[-1], f"batch_y shape: {batch_y.shape}, outputs shape: {outputs.shape}, exo {self.args.exo}, exo_future {self.args.exo_future}, shapes do not match"

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                DTW_error_per_signal = self.compute_dtw(outputs, batch_y)
                print(DTW_error_per_signal.shape)

                for i, th in enumerate(mse_threshold):
                    nice_signals = DTW_error_per_signal < th

                    b = nice_signals.shape[0]
                    s = nice_signals.shape[1]
                    cls_c[i] += nice_signals.sum(axis=0)
                    far_c[i] += b - nice_signals.sum(axis=0)
            

        all_c = [c + f for c, f in zip(cls_c, far_c)]
        print(all_c)
        print(cls_c)
        print(far_c)
        cls_2all =  [[a/b for a, b in zip(cls_c_i, all_c_i)] for cls_c_i, all_c_i in zip(cls_c, all_c)]
        far_2all =  [[a/b for a, b in zip(far_c_i, all_c_i)] for far_c_i, all_c_i in zip(far_c, all_c)]
        print(('DTW Error, test all, MAE threshold: {}, close samples: {} {}%, far samples:{} {}%, all:{}'.format(mse_threshold, cls_c, cls_2all, far_c, far_2all, all_c)))
    
        
        acc = []
        auc_list = []
        auc_normalized_list = []
        for i in range(out_signal_n):
            path_fig = os.path.join(folder_path, f'accuracy_error_plot{i}.pdf')
            path_vec = os.path.join(folder_path, f'accuracy_error_plot{i}.npy')
            print(path_fig)
            acc_i = [cls_i[i] for cls_i in cls_2all]
            auc, auc_normalized = visual_acc(mse_threshold, acc_i, path_fig)
            auc_list.append(auc)
            auc_normalized_list.append(auc_normalized)
            np.save(path_vec, np.array([mse_threshold, acc_i]))
        
        
        f = open("result.txt", 'a')
        f.write('Test End Time: {}\n'.format(time.strftime("%Y.%m.%d,%H:%M:%S", time.localtime())))
        f.write('AUC:{}, AUC normalized:{}'.format(auc, auc_normalized))
        f.write('\n')
        f.write('AUC per signal:{}'.format(auc_list))
        f.write('\n')
        f.write('AUC normalized per signal:{}'.format(auc_normalized_list))
        f.write('\n')
        f.write('Folder path: {}'.format(folder_path))
        f.write('\n')
        f.write('\n')
        f.close()
        return

    # def predict(self, setting, load=False):

    # TODO: implement predict method
    # no ground truth for prediction

    #     return
