from distutils.util import subst_vars
import logging
import time
from pathlib import Path
from typing import Tuple
import pandas as pd
import torch
import torch.optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
# from utils import PLCC,SROCC,RMSE,MAE
from common import AverageMeter, Transform
from dataset import AVADataset1
from emd_loss import EDMLoss
import models_emd
from argparse import ArgumentParser
import os
from visdom import Visdom
import numpy as np
import functools
from scipy import stats
logger = logging.getLogger(__file__)
import torch.nn.functional as F
from scipy.stats import wasserstein_distance
from collections import Counter
import csv
#指标：
# MSE = np.mean(np.square(y - y_hat))
# RMSE = np.sqrt(np.mean(np.square(y - y_hat)))
# MAE = np.mean(np.abs(y-y_hat))


#path_to_save_csv='/myDockerShare/zys/emotion/csv'
def logging(s, log_path, print_=True, log_=True):
    if print_:
        print(s)
    if log_:
        with open(log_path, 'a+') as f_log:
            f_log.write(s + '\n')
def get_logger(log_path, **kwargs):
    return functools.partial(logging, log_path=log_path, **kwargs)

logging = get_logger('/myDockerShare/zys/aes-emotion/experiment/log_emd.txt')

def get_dataloaders(
    path_to_save_csv: Path, batch_size: int, num_workers: int
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    transform = Transform()

    train_ds = AVADataset1(os.path.join(path_to_save_csv , "train.csv"), '/myDockerShare/zys/ava/aes/train', transform.train_transform)
    val_ds = AVADataset1(os.path.join(path_to_save_csv , "test.csv"), '/myDockerShare/zys/ava/aes/val', transform.val_transform)
    test_ds = AVADataset1(os.path.join(path_to_save_csv , "try.csv"), '/myDockerShare/zys/ava/aes-im', transform.val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return train_loader, val_loader, test_loader



class Trainer:
    def __init__(
        self,
        *,
        path_to_save_csv: Path,
        # path_to_images: Path,
        num_epoch: int,
        # model_type: str,
        num_workers: int,
        batch_size: int,
        init_lr: float,
        lr_ratio: float,
        weight_decay: float,
        experiment_dir: Path,
        # drop_out: float,
        # optimizer_type: str,
    ):

        train_loader, val_loader, test_loader= get_dataloaders(
            path_to_save_csv=path_to_save_csv,
            # path_to_images=path_to_images,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        predict_model = torch.load('/myDockerShare/zys/aes-emotion/experiment/fusion4.pth')
        self.model_hyper= models_emd.HyperNet(16, 112, 224, 112, 98, 49, 10, 7).cuda()
        net_dict = self.model_hyper.state_dict()
        state_dict = {k: v for k, v in predict_model.items() if k in net_dict.keys()}
        net_dict.update(state_dict)  # 将预训练参数更新到新的网络层
        self.model_hyper.load_state_dict(net_dict)
        self.model_hyper.train(True)
        backbone_params = list(map(id, self.model_hyper.res.parameters()))
        self.hypernet_params = filter(lambda p: id(p) not in backbone_params, self.model_hyper.parameters())
        self.lr = init_lr
        self.lrratio =lr_ratio
        self.weight_decay = weight_decay
        paras = [{'params': self.hypernet_params, 'lr': self.lr * self.lrratio},
                 {'params': self.model_hyper.res.parameters(), 'lr': self.lr}
                 ]
        self.optimizer  = torch.optim.Adam(paras, weight_decay=self.weight_decay)

        self.scheduler=torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer,step_size=5,gamma=0.1)
        self.criterion = EDMLoss().to(self.device)
        # self.criterion = torch.nn.MSELoss().cuda()
        experiment_dir.mkdir(exist_ok=True, parents=True)
        self.experiment_dir = experiment_dir
        self.writer = SummaryWriter(str(os.path.join(experiment_dir , "logs")))
        self.num_epoch = num_epoch
        self.global_train_step = 0
        self.global_val_step = 0
        self.print_freq = 100


    def train(self):
        # plcc=PLCC()
        # srocc=SROCC()
        # rmse=RMSE()
        
        
        best_srcc=0
        emd = 0
        sub1 = []
        obj1 = []
        sub_s = []
        obj_s = []
        sub_l=[]
        obj_l=[]
        train_losses = AverageMeter()
        total_iter = len(self.train_loader.dataset) // self.train_loader.batch_size
        for t in range(0, self.num_epoch ):
            train_acc=0
            # path_to_model_state=os.path.join(self.experiment_dir ,"best_state.pth")
            # best_state = torch.load('/myDockerShare/zys/aes-emotion/experiment/fusion4.pth')
            # self.model_hyper.load_state_dict(best_state["state_dict1"])
          
            for idx, (x, y) in enumerate(self.train_loader):
                s = time.monotonic()
                # self.model_hyper.load_state_dict(best_state["state_dict1"])
                self.model_hyper.train()
                x = x.to(self.device)
                y = y.to(self.device)
                for i in range(len(y)):
                    sub= y[i][0]*1+y[i][1]*2+y[i][2]*3+y[i][3]*4+y[i][4]*5+y[i][5]*6+y[i][6]*7+y[i][7]*8+y[i][8]*9+y[i][9]*10
                    sub_s = sub_s+sub.flatten().cpu().detach().numpy().tolist()
                for item in sub_s:
                    if item<5:
                        item=0
                        sub_l.append(item)
                    else:
                        item=1
                        sub_l.append(item)
                # print('y:',y)
                paras = self.model_hyper(x)  # 'paras' contains the network weights conveyed to target network
                model = models_emd.TargetNet(paras).cuda()
                # model.load_state_dict(best_state["state_dict"])
                model.train()
                # for param in model.parameters():
                #         param.requires_grad = True
                y_pred = model(paras['target_in_vec'])
                for i in range(len(y_pred )):
                    obj= y_pred[i][0]*1+y_pred[i][1]*2+y_pred[i][2]*3+y_pred[i][3]*4+y_pred[i][4]*5+y_pred[i][5]*6+y_pred[i][6]*7+y_pred[i][7]*8+y_pred[i][8]*9+y_pred[i][9]*10
                    obj_s = obj_s+obj.flatten().cpu().detach().numpy().tolist()
                # print('pred',y_pred)
                for item in obj_s:
                    if item<5:
                        item=0
                        obj_l.append(item)
                    else:
                        item=1
                        obj_l.append(item)
                obj1 = obj1 + y_pred.cpu().tolist()
                sub1 = sub1 + y.cpu().tolist()
                loss =self.criterion(y_pred, y)
                # loss = self.criterion(y_pred.squeeze(), y.float().detach())
                self.optimizer.zero_grad()

                loss.backward()

                self.optimizer.step()
                train_losses.update(loss.item(), x.size(0))

                
                self.writer.add_scalar("train/current_loss", train_losses.val, self.global_train_step)
                self.writer.add_scalar("train/avg_loss", train_losses.avg, self.global_train_step)
                
                self.global_train_step += 1

                e = time.monotonic()
                if idx % self.print_freq:
                    log_time = self.print_freq * (e - s)
                    eta = ((total_iter - idx) * log_time) / 60.0
                    print(f"iter #[{idx}/{total_iter}] " f"loss = {loss:.3f} " f"time = {log_time:.2f} " f"eta = {eta:.2f}")
            X1 = pd.Series(sub1)
            # # print('x1:',X1)
            Y1 = pd.Series(obj1)
            # # print('y1:',Y1)
            for i,item in enumerate(zip(X1,Y1)):
                emd+=wasserstein_distance(item[0],item[1])
            # #     # print('X1:',item[0],'Y1:',item[1])
            train_emd=emd/len(X1)
            X_s = pd.Series(sub_s)
            Y_s= pd.Series(obj_s)
            train_plcc = X_s.corr(Y_s, method="pearson")
            train_srcc = X_s.corr(Y_s, method="spearman")
            train_mse= np.mean(np.square(np.array(obj_s)-np.array(sub_s)))
            
            v = list(map(lambda x: x[0]-x[1], zip(sub_l, obj_l)))
            count = Counter(v)
            acc=count[0]/len(sub_l)
            print('Train==>','epoch:',t+1,'train_loss:',train_losses.avg,'PLCC:',train_plcc,'SRCC:',train_srcc,'MSE:',train_mse,'Emd:',train_emd,'ACC:',acc)

            # viz.line(X=np.array([t+1]), Y=np.array([train_losses.avg]), win=win1, update='append')
            # viz.line(X=np.array([t+1]), Y=np.array([train_acc]), win=win3, update='append')
  
            validate_losses,val_plcc,val_srcc,val_mse,val_emd,val_acc= self.validate()
            # self.scheduler.step(metrics=validate_losses)
            # viz.line(X=np.array([t+1]), Y=np.array([validate_losses]), win=win2, update='append')
            self.scheduler.step()

            logging("VAL Results - Epoch: {} PLCC:{:.4f} SRCC:{:.4f} MSE:{:.4f} EMD: {:.4f} ACC:{:.4f} "
                    .format(t+1, val_plcc,val_srcc,val_mse,val_emd,val_acc))
            if val_srcc > best_srcc:
                best_srcc = val_srcc
                best_state = {
                    "state_dict": model.state_dict(),
                    "state_dict1": self.model_hyper.state_dict(),
                    "epoch": t,
                    "best_loss": validate_losses,
                }
                torch.save(best_state, os.path.join(self.experiment_dir ,"best_state.pth"))
        
            print('epoch:',t + 1, 'val_srcc:',val_srcc)
            
            test_plcc,test_srcc,test_mse,test_emd,test_acc= self.test()
            logging("TEST Results - Epoch: {} PLCC:{:.4f} SRCC:{:.4f} MSE:{:.4f} EMD: {:.4f} ACC:{:.4f} "
                    .format(t+1, test_plcc,test_srcc,test_mse,test_emd,test_acc))

            # viz.line(X=np.array([t+1]), Y=np.array([test_acc]), win=win4, update='append')
                 
                # Update optimizer
            # lr = self.lr / pow(10, (t // 6))
            # if t > 8:
            #     self.lrratio = 1
            # self.paras = [{'params': self.hypernet_params, 'lr': lr * self.lrratio},
            #                 {'params': self.model_hyper.res.parameters(), 'lr': self.lr}
            #                 ]
            # self.solver = torch.optim.Adam(self.paras, weight_decay=self.weight_decay)

          
        return best_srcc

    def validate(self):
        self.model_hyper.train(False)
        validate_losses = AverageMeter()
    
        val_acc=0
        emd = 0
        obj_v=[]
        sub_v=[]
        sub_vs = []
        obj_vs = []
        sub_vl=[]
        obj_vl=[]
        with torch.no_grad():
            for idx, (x, y) in enumerate(self.val_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                for i in range(len(y)):
                    sub= y[i][0]*1+y[i][1]*2+y[i][2]*3+y[i][3]*4+y[i][4]*5+y[i][5]*6+y[i][6]*7+y[i][7]*8+y[i][8]*9+y[i][9]*10
                    sub_vs = sub_vs+sub.flatten().cpu().detach().numpy().tolist()
                for item in sub_vs :
                    if item<5:
                        item=0
                        sub_vl.append(item)
                    else:
                        item=1
                        sub_vl.append(item)
                paras = self.model_hyper(x)  # 'paras' contains the network weights conveyed to target network
                model = models_emd.TargetNet(paras).cuda()
                model.train(False)
                y_pred = model(paras['target_in_vec'])
                for i in range(len(y_pred )):
                    obj= y_pred[i][0]*1+y_pred[i][1]*2+y_pred[i][2]*3+y_pred[i][3]*4+y_pred[i][4]*5+y_pred[i][5]*6+y_pred[i][6]*7+y_pred[i][7]*8+y_pred[i][8]*9+y_pred[i][9]*10
                    obj_vs = obj_vs+obj.flatten().cpu().detach().numpy().tolist()
                for item in obj_vs:
                    if item<5:
                        item=0
                        obj_vl.append(item)
                    else:
                        item=1
                        obj_vl.append(item)
                loss =self.criterion(y_pred, y)
                validate_losses.update(loss.item(), x.size(0))

                obj_v = obj_v + y_pred.cpu().tolist()
                sub_v = sub_v + y.cpu().tolist()
                self.writer.add_scalar("val/current_loss", validate_losses.val, self.global_val_step)
                self.writer.add_scalar("val/avg_loss", validate_losses.avg, self.global_val_step)
                self.global_val_step += 1
        X1 = pd.Series(sub_v)
        Y1 = pd.Series(obj_v)
        for idx,item in enumerate(zip(X1,Y1)):
            emd+=wasserstein_distance(item[0],item[1])

        val_emd=emd/len(X1)

        X_vs = pd.Series(sub_vs)
        Y_vs= pd.Series(obj_vs)
        val_plcc = X_vs.corr(Y_vs, method="pearson")
        val_srcc = X_vs.corr(Y_vs, method="spearman")
        val_mse= np.mean(np.square(np.array(obj_vs)-np.array(sub_vs)))
        vv = list(map(lambda x: x[0]-x[1], zip(sub_vl, obj_vl)))
        count = Counter(vv)
        acc=count[0]/len(sub_vl)
        return validate_losses.avg,val_plcc,val_srcc,val_mse,val_emd,acc

    def test(self):
        """Testing"""
        obj_t=[]
        sub_t=[]
        sub_ts = []
        obj_ts = []
        # sub_tl=[]
        # obj_tl=[]
        emd = 0
        path_to_model_state=os.path.join(self.experiment_dir ,"fusion4.pth")
        best_state = torch.load(path_to_model_state)
        self.model_hyper.eval()
        with torch.no_grad():
            test_acc=0
            for idx, (img, label) in enumerate(self.test_loader):
                # Data.
                img = img.to(self.device)
                label = label.to(self.device)
                # for i in range(len(label)):
                #     sub= label[i][0]*1+label[i][1]*2+label[i][2]*3+label[i][3]*4+label[i][4]*5+label[i][5]*6+label[i][6]*7+label[i][7]*8+label[i][8]*9+label[i][9]*10
                #     sub_ts = sub_ts+sub.flatten().cpu().detach().numpy().tolist()
                # for item in sub_ts:
                #     if item<5:
                #         item=0
                #         sub_tl.append(item)
                #     else:
                #         item=1
                #         sub_tl.append(item)
                paras = self.model_hyper(img)
                model_target = models_emd.TargetNet(paras).cuda()
                model_target.load_state_dict(best_state["state_dict"])
                model_target.eval()
                y_pred = model_target(paras['target_in_vec'])
                # for i in range(len(y_pred)):
                #     obj= y_pred[i][0]*1+y_pred[i][1]*2+y_pred[i][2]*3+y_pred[i][3]*4+y_pred[i][4]*5+y_pred[i][5]*6+y_pred[i][6]*7+y_pred[i][7]*8+y_pred[i][8]*9+y_pred[i][9]*10
                #     obj_ts = obj_ts+obj.flatten().cpu().detach().numpy().tolist()
                # for item in obj_ts:
                #     if item<5:
                #         item=0
                #         obj_tl.append(item)
                #     else:
                #         item=1
                #         obj_tl.append(item)
                obj_t = obj_t + y_pred.cpu().tolist()
                sub_t = sub_t + label.cpu().tolist()
                
        print('y:',sub_t)
        print('y_pred:',obj_t)
        # print('y:',sub_ts)
        # print('y_pred:',obj_ts)
        # X1 = pd.Series(sub_t)
        # Y1 = pd.Series(obj_t)
                  
        # for idx,item in enumerate(zip(X1,Y1)):
        #     emd+=wasserstein_distance(item[0],item[1])

        # test_emd=emd/len(X1)
        # X_ts = pd.Series(sub_ts)
        # Y_ts= pd.Series(obj_ts)
        # test_plcc = X_ts.corr(Y_ts, method="pearson")
        # test_srcc = X_ts.corr(Y_ts, method="spearman")
        # test_mse= np.mean(np.square(np.array(obj_ts)-np.array(sub_ts)))
        # # print('test==','EMD:', test_emd)
        # tv = list(map(lambda x: x[0]-x[1], zip(sub_tl, obj_tl)))
        # count = Counter(tv)
        # acc=count[0]/len(sub_tl)
    
        return obj_t,sub_t

if __name__=='__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    parser=ArgumentParser(description='NIMA')
    parser.add_argument('--path_to_save_csv',type=str,default='/myDockerShare/zys/ava/csv',help='data-label')
    parser.add_argument('--experiment_dir',type=Path,default='/myDockerShare/zys/aes-emotion/experiment',help='experiment_dir')
    parser.add_argument('--num_epoch',type=int,default=50, help='eopch')
    parser.add_argument('--model_type',type=str,default='resnet50',help=' model_type')
    parser.add_argument('--optimizer_type',type=str,default='adam',help=' optimizer_type')
    parser.add_argument('--num_workers',type=int,default=8, help='num_workers')
    parser.add_argument('--batch_size',type=int,default=32, help='batch_size')
    parser.add_argument('--init_lr',type=float,default=2e-5, help='init_lr')
    parser.add_argument('--lr_ratio',type=float,default=10,help='lr_ratio')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--drop_out',type=float,default=0.75, help='drop_out')
    args=parser.parse_args()
    slover=Trainer(path_to_save_csv=args.path_to_save_csv,num_epoch=args.num_epoch,num_workers=args.num_workers,batch_size=args.batch_size,init_lr=args.init_lr,
                lr_ratio=args.lr_ratio,weight_decay=args.weight_decay,experiment_dir=args.experiment_dir)
    best_srcc= slover.test()
    # print('best:',best_srcc)
    # main(args.path_to_save_csv,args.num_epoch,args.model_type,args.num_workers,args.batch_size,args.init_lr,
    #         args.experiment_dir,args.drop_out,args.optimizer_type)

